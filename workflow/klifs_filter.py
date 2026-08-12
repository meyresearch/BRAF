"""KLIFS-based DFG/APE motif filter for kinase chain PDBs.

This module replaces the previous MSA-based motif filter for chains that are
**curated by KLIFS** (https://klifs.net/). For each (PDB, chain) pair in the
extracted InterPro chains directory we:

1. Query the KLIFS REST API to determine membership and obtain a ``structure_ID``.
2. Query the KLIFS "match_residues" endpoint to map the canonical KLIFS pocket
   positions to PDB residue numbers. KLIFS pocket position 81 = DFG-Asp.
3. Find the next ``APE`` triplet in the chain's sequence downstream of DFG.
4. Define the activation loop as the residues between DFG-end and APE-start.
5. Copy the chain PDB to ``Results/motif_filtered_chains/`` and its paired
   small-molecule file to ``Results/motif_filtered_small_molecules/``.
6. Write the activation-loop sequence (gaps stripped) to
   ``Results/activation_loop_sequences.tsv`` so the Tukey loop-length filter
   in :class:`workflow.ca_stripper.OutlierStripper` runs unchanged.

Chains that are **not** in KLIFS are listed in
``Results/KLIFS/non_klifs_chain_basenames.txt``. After :meth:`KLIFSOverlap.run_filter`,
``Results/KLIFS/hmmer_input_basenames.txt`` unions those with chains that **are** in
KLIFS but failed the DFG/APE step, so :mod:`workflow.hmmer_diagnostics` can scan them
with ``hmmscan``.

The KLIFS HTTP responses are cached to TSV so the notebook can be re-run
without re-hitting the API. A **full-database** ``(PDB, chain)`` table for the
Venn diagram is built by :meth:`KLIFSOverlap.build_full_klifs_catalog` (slow once,
then cached).
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import time
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from tqdm.auto import tqdm

from workflow.chain_basenames import load_excluded_pseudokinase_basenames

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "The 'requests' package is required for KLIFS API access. "
        "Install it with: pip install requests"
    ) from exc

KLIFS_API = "https://klifs.net/api"

DEFAULT_INPUT_DIR = "Results/InterPro_protein_chains/"
DEFAULT_SMALL_MOLECULE_DIR = "Results/InterPro_protein_small_molecules/"
DEFAULT_KLIFS_DIR = "Results/KLIFS"
DEFAULT_TARGET_PROTEIN_DIR = "Results/motif_filtered_chains/"
DEFAULT_TARGET_LIGAND_DIR = "Results/motif_filtered_small_molecules/"
DEFAULT_LOOP_TSV = "Results/activation_loop_sequences.tsv"
DEFAULT_PSEUDO_FILE = "Results/excluded_pseudokinase_basenames.txt"

# KLIFS canonical pocket position for DFG-Asp.
KLIFS_POS_DFG_ASP = 81


def _clear_and_make_dir(dir_path: str) -> None:
    """Clear *dir_path* contents or create it (same behaviour as ``utilities.clear_and_make``).

    Kept local so :meth:`KLIFSOverlap.run_filter` does not import ``workflow.utilities``,
    which pulls optional heavy dependencies such as ``mdtraj``.
    """
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    else:
        os.makedirs(dir_path)

# Patterns for our chain basenames.
_CHAIN_LONG = re.compile(r"^([A-Za-z0-9]{4})_chain([A-Za-z0-9]+)$")  # 6UAN_chainD
_CHAIN_SHORT = re.compile(r"^([A-Za-z0-9]{4})_([A-Za-z0-9]+)$")      # 6G9D_A


def _parse_basename(basename: str) -> Optional[Tuple[str, str]]:
    """Return (pdb_id, chain_id) parsed from a chain basename, or ``None``."""
    m = _CHAIN_LONG.match(basename) or _CHAIN_SHORT.match(basename)
    if not m:
        return None
    return m.group(1).upper(), m.group(2)


def _chain_id_prefix(filename: str) -> str:
    """Return the ``PDBID_CHAIN`` prefix shared by paired protein/ligand files."""
    stem, _ = os.path.splitext(filename)
    parts = stem.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem


class KLIFSOverlap:
    """Inventory which chains are in KLIFS, plot a Venn diagram, and run the
    DFG/APE motif filter on KLIFS-curated chains."""

    def __init__(
        self,
        input_dir: str = DEFAULT_INPUT_DIR,
        small_molecule_dir: str = DEFAULT_SMALL_MOLECULE_DIR,
        klifs_dir: str = DEFAULT_KLIFS_DIR,
        target_protein_dir: str = DEFAULT_TARGET_PROTEIN_DIR,
        target_ligand_dir: str = DEFAULT_TARGET_LIGAND_DIR,
        loop_tsv: str = DEFAULT_LOOP_TSV,
        pseudo_basenames_path: str = DEFAULT_PSEUDO_FILE,
        request_delay_s: float = 0.0,
        batch_size: int = 50,
        timeout_s: int = 60,
    ):
        self.input_dir = input_dir
        self.small_molecule_dir = small_molecule_dir
        self.klifs_dir = klifs_dir
        self.target_protein_dir = target_protein_dir
        self.target_ligand_dir = target_ligand_dir
        self.loop_tsv = loop_tsv
        self.pseudo_basenames_path = pseudo_basenames_path
        self.request_delay_s = request_delay_s
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        os.makedirs(self.klifs_dir, exist_ok=True)

    # ── output paths ─────────────────────────────────────────────────────
    @property
    def inventory_tsv(self) -> str:
        return os.path.join(self.klifs_dir, "klifs_chain_inventory.tsv")

    @property
    def klifs_basenames_txt(self) -> str:
        return os.path.join(self.klifs_dir, "klifs_chain_basenames.txt")

    @property
    def non_klifs_basenames_txt(self) -> str:
        return os.path.join(self.klifs_dir, "non_klifs_chain_basenames.txt")

    @property
    def hmmer_input_basenames_txt(self) -> str:
        """Basenames for HMMER: not-in-KLIFS ∪ KLIFS motif filter skips (written by :meth:`run_filter`)."""
        return os.path.join(self.klifs_dir, "hmmer_input_basenames.txt")

    @property
    def venn_png(self) -> str:
        return os.path.join(self.klifs_dir, "venn_interpro_vs_klifs.png")

    @property
    def structure_inventory_tsv(self) -> str:
        return os.path.join(self.klifs_dir, "klifs_structure_inventory.tsv")

    @property
    def structure_venn_png(self) -> str:
        return os.path.join(self.klifs_dir, "venn_interpro_vs_klifs_structures.png")

    @property
    def structure_venn_summary_json(self) -> str:
        """Counts for :meth:`plot_venn_structures` (written by :meth:`build_structure_inventory`)."""
        return os.path.join(self.klifs_dir, "klifs_structure_venn_summary.json")

    @property
    def venn_summary_json(self) -> str:
        """Counts for :meth:`plot_venn` (written by :meth:`build_inventory`)."""
        return os.path.join(self.klifs_dir, "klifs_venn_summary.json")

    @property
    def klifs_pdb_list_cache_tsv(self) -> str:
        """Last ``structures_pdb_list`` response (used to draw KLIFS \\ InterPro in :meth:`plot_venn`)."""
        return os.path.join(self.klifs_dir, "klifs_structures_pdb_list_response.tsv")

    @property
    def klifs_full_catalog_tsv(self) -> str:
        """Deduplicated *(PDB, chain)* rows from KLIFS ``structures_list`` across all kinases."""
        return os.path.join(self.klifs_dir, "klifs_full_catalog_pdb_chain.tsv")

    @property
    def klifs_full_catalog_summary_json(self) -> str:
        return os.path.join(self.klifs_dir, "klifs_full_catalog_summary.json")

    @property
    def klifs_failures_txt(self) -> str:
        return os.path.join(self.klifs_dir, "klifs_filter_failures.txt")

    @property
    def filter_summary_json(self) -> str:
        return os.path.join(self.klifs_dir, "klifs_filter_summary.json")

    # ── KLIFS REST helpers ───────────────────────────────────────────────
    def _klifs_structures_for_pdbs(self, pdb_codes: List[str]) -> pd.DataFrame:
        """Query ``/structures_pdb_list`` in batches; return a DataFrame of hits."""
        if not pdb_codes:
            return pd.DataFrame()
        rows: List[dict] = []
        for i in tqdm(
            range(0, len(pdb_codes), self.batch_size),
            desc="KLIFS structures_pdb_list",
        ):
            batch = pdb_codes[i:i + self.batch_size]
            try:
                r = requests.get(
                    f"{KLIFS_API}/structures_pdb_list",
                    params={"pdb-codes": ",".join(batch)},
                    timeout=self.timeout_s,
                )
            except Exception as exc:
                print(f"  KLIFS batch failed ({batch[0]}..): {exc}")
                continue
            if r.status_code == 404:
                continue
            if r.status_code != 200:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            rows.extend(data if isinstance(data, list) else [])
            if self.request_delay_s:
                time.sleep(self.request_delay_s)
        return pd.DataFrame(rows)

    def _klifs_match_residues(self, structure_id: int) -> Optional[pd.DataFrame]:
        """Query ``/interactions_match_residues`` for KLIFS-pos → PDB residue.

        Note: ``/interactions/match_residues`` (slash) returns HTTP 400 on the public
        server; the underscore route is the one documented in the KLIFS Swagger UI.
        """
        try:
            r = requests.get(
                f"{KLIFS_API}/interactions_match_residues",
                params={"structure_ID": structure_id},
                timeout=self.timeout_s,
            )
        except Exception:
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except Exception:
            return None
        return pd.DataFrame(data)

    # ── Step 1: inventory ────────────────────────────────────────────────
    def _list_chain_files(self) -> List[Tuple[str, str, str]]:
        """List (basename, pdb_id, chain_id) for every PDB file in ``input_dir``."""
        out = []
        for f in sorted(os.listdir(self.input_dir)):
            if not f.lower().endswith(".pdb"):
                continue
            base = os.path.splitext(f)[0]
            parsed = _parse_basename(base)
            if parsed is None:
                continue
            out.append((base, parsed[0], parsed[1]))
        return out

    @staticmethod
    def _our_pairs_from_inventory(inv: pd.DataFrame) -> set:
        """Distinct (PDB, chain) from the InterPro-derived inventory."""
        return set(
            zip(
                inv["pdb_id"].astype(str).str.upper(),
                inv["chain_id"].astype(str).str.upper(),
            )
        )

    @staticmethod
    def _pairs_from_klifs_pdb_response(kdf: pd.DataFrame) -> set:
        """Distinct (PDB, chain) from a ``structures_pdb_list``-style DataFrame."""
        if kdf.empty:
            return set()
        col_pdb = "pdb" if "pdb" in kdf.columns else None
        col_chain = "chain" if "chain" in kdf.columns else None
        if col_pdb is None or col_chain is None:
            return set()
        return set(
            zip(
                kdf[col_pdb].astype(str).str.upper(),
                kdf[col_chain].astype(str).str.upper(),
            )
        )

    def _write_klifs_pdb_list_cache(self, klifs_df: pd.DataFrame) -> None:
        """Persist KLIFS response so :meth:`plot_venn` can show KLIFS \\ InterPro without re-querying."""
        os.makedirs(self.klifs_dir, exist_ok=True)
        if klifs_df.empty:
            pd.DataFrame(columns=["pdb", "chain", "structure_ID"]).to_csv(
                self.klifs_pdb_list_cache_tsv, sep="\t", index=False
            )
        else:
            klifs_df.to_csv(self.klifs_pdb_list_cache_tsv, sep="\t", index=False)
        print(f"  KLIFS PDB-list cache → {self.klifs_pdb_list_cache_tsv}")

    def build_full_klifs_catalog(self, force: bool = False) -> pd.DataFrame:
        """Build a deduplicated table of every *(PDB, chain)* KLIFS annotates (HTTP API).

        Walks ``/kinase_information`` to obtain all ``kinase_ID`` values, then calls
        ``/structures_list`` per kinase. Many kinases return HTTP 400 (no structures);
        those are skipped. Rows are **deduplicated** on *(PDB, chain)* (first hit kept).

        This is slow on first run (~1100 requests, order minutes) but cached to
        ``klifs_full_catalog_pdb_chain.tsv`` so later runs read from disk. The Venn
        diagram’s **KLIFS-only** region uses this file when present; otherwise it falls
        back to the PDB-scoped ``structures_pdb_list`` cache from :meth:`build_inventory`.
        """
        if (not force) and os.path.isfile(self.klifs_full_catalog_tsv):
            print(f"Using cached full KLIFS catalog: {self.klifs_full_catalog_tsv}")
            return pd.read_csv(self.klifs_full_catalog_tsv, sep="\t")

        t0 = time.time()
        r = requests.get(
            f"{KLIFS_API}/kinase_information",
            timeout=max(self.timeout_s, 120),
        )
        r.raise_for_status()
        kinases = r.json()
        if not isinstance(kinases, list):
            raise RuntimeError("Unexpected /kinase_information response (expected a list).")
        kid_list = [int(k["kinase_ID"]) for k in kinases]

        best: Dict[Tuple[str, str], Dict[str, object]] = {}
        raw_rows = 0
        n_ok = 0
        for kid in tqdm(kid_list, desc="KLIFS full catalog (structures_list)"):
            try:
                rs = requests.get(
                    f"{KLIFS_API}/structures_list",
                    params={"kinase_ID": kid},
                    timeout=self.timeout_s,
                )
            except Exception as exc:
                print(f"  skip kinase_ID={kid}: {exc}")
                continue
            if rs.status_code == 400:
                continue
            if rs.status_code != 200:
                continue
            try:
                data = rs.json()
            except Exception:
                continue
            if not isinstance(data, list) or not data:
                continue
            n_ok += 1
            for d in data:
                raw_rows += 1
                pdb_u = str(d.get("pdb", "")).upper()
                ch_u = str(d.get("chain", "")).upper()
                if not pdb_u:
                    continue
                key = (pdb_u, ch_u)
                if key not in best:
                    best[key] = {
                        "pdb": pdb_u,
                        "chain": ch_u,
                        "structure_ID": d.get("structure_ID"),
                        "kinase_ID": kid,
                    }
            if self.request_delay_s:
                time.sleep(self.request_delay_s)

        catalog = pd.DataFrame(list(best.values()))
        if not catalog.empty:
            catalog = catalog.sort_values(["pdb", "chain"]).reset_index(drop=True)
        os.makedirs(self.klifs_dir, exist_ok=True)
        catalog.to_csv(self.klifs_full_catalog_tsv, sep="\t", index=False)

        wall = round(time.time() - t0, 1)
        summary = {
            "n_kinase_information_rows": len(kid_list),
            "n_kinases_with_structures_http200": n_ok,
            "n_kinases_skipped_http400_or_empty": len(kid_list) - n_ok,
            "n_raw_structure_rows_seen": raw_rows,
            "n_unique_pdb_chain_pairs": int(len(catalog)),
            "wall_time_s": wall,
            "catalog_path": self.klifs_full_catalog_tsv,
            "note": (
                "Union of /structures_list over kinase_ID from /kinase_information. "
                "HTTP 400 means KLIFS has no structures for that kinase entry. "
                "This is the practical KLIFS 'full database' surface exposed by the API "
                "for Venn purposes (unique PDB chain IDs)."
            ),
        }
        with open(self.klifs_full_catalog_summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        print(
            f"Wrote {len(catalog)} unique (PDB, chain) rows → {self.klifs_full_catalog_tsv}"
        )
        print(f"Summary → {self.klifs_full_catalog_summary_json} ({wall}s)")
        return catalog

    def build_inventory(self, force: bool = False) -> pd.DataFrame:
        """Build a per-chain inventory of KLIFS membership.

        Writes ``klifs_chain_inventory.tsv``, ``klifs_chain_basenames.txt`` and
        ``non_klifs_chain_basenames.txt``. Re-uses the cached TSV when present
        (pass ``force=True`` to re-query the API).
        """
        if (not force) and os.path.isfile(self.inventory_tsv):
            print(f"Reusing cached inventory: {self.inventory_tsv}")
            if not os.path.isfile(self.venn_summary_json) or not os.path.isfile(
                self.klifs_pdb_list_cache_tsv
            ):
                print(
                    f"  Note: {self.klifs_pdb_list_cache_tsv} and/or "
                    f"{self.venn_summary_json} missing — run build_inventory(force=True) "
                    "once so the Venn diagram can show KLIFS chains not in this extract."
                )
            return pd.read_csv(self.inventory_tsv, sep="\t")

        triples = self._list_chain_files()
        unique_pdbs = sorted({pdb for _, pdb, _ in triples})
        print(
            f"Scanning {len(triples)} chain files "
            f"({len(unique_pdbs)} unique PDB IDs) against KLIFS"
        )

        klifs_df = self._klifs_structures_for_pdbs(unique_pdbs)
        self._write_klifs_pdb_list_cache(klifs_df)

        if klifs_df.empty:
            print("  KLIFS returned no matching structures.")
            structure_id_lookup: Dict[Tuple[str, str], List[int]] = {}
        else:
            klifs_df["pdb"] = klifs_df["pdb"].astype(str).str.upper()
            klifs_df["chain"] = klifs_df["chain"].astype(str).str.upper()
            structure_id_lookup = defaultdict(list)
            for _, row in klifs_df.iterrows():
                key = (row["pdb"], row["chain"])
                structure_id_lookup[key].append(int(row["structure_ID"]))

        rows = []
        for base, pdb, ch in triples:
            ch_u = str(ch).upper()
            sids = structure_id_lookup.get((pdb, ch_u), [])
            rows.append({
                "basename": base,
                "pdb_id": pdb,
                "chain_id": ch,
                "in_klifs": bool(sids),
                "klifs_structure_ids": ",".join(str(s) for s in sids),
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.inventory_tsv, sep="\t", index=False)

        in_klifs = int(df["in_klifs"].sum())
        our_pairs = self._our_pairs_from_inventory(df)
        klifs_pairs = self._pairs_from_klifs_pdb_response(klifs_df)
        n_klifs_only_same_pdbs = len(klifs_pairs - our_pairs)
        venn_summary = {
            "n_interpro_chain_files": len(triples),
            "n_unique_pdbs_queried_klifs": len(unique_pdbs),
            "n_interpro_chains_in_klifs": in_klifs,
            "n_interpro_chains_not_in_klifs": len(df) - in_klifs,
            "n_klifs_pairs_on_queried_pdbs_not_in_interpro_extract": int(
                n_klifs_only_same_pdbs
            ),
            "n_klifs_api_structure_rows": int(len(klifs_df)),
            "explanation": (
                "n_klifs_pairs_on_queried_pdbs_not_in_interpro_extract counts distinct "
                "(PDB, chain) from structures_pdb_list for your PDB IDs only, minus your "
                "extract. For the **full** KLIFS-vs-InterPro Venn crescent, run "
                "build_full_klifs_catalog() which caches all (pdb, chain) from "
                "structures_list across kinases."
            ),
        }
        with open(self.venn_summary_json, "w", encoding="utf-8") as fh:
            json.dump(venn_summary, fh, indent=2)
        print(f"  Venn counts       →  {self.venn_summary_json}")

        print(
            f"  in KLIFS    : {in_klifs} / {len(df)} chains"
            f"  →  {self.inventory_tsv}"
        )
        df.loc[df["in_klifs"], "basename"].to_csv(
            self.klifs_basenames_txt, index=False, header=False
        )
        df.loc[~df["in_klifs"], "basename"].to_csv(
            self.non_klifs_basenames_txt, index=False, header=False
        )
        print(f"  KLIFS basenames     →  {self.klifs_basenames_txt}")
        print(f"  non-KLIFS basenames →  {self.non_klifs_basenames_txt}")
        return df

    def _has_nonempty_full_klifs_catalog(self) -> bool:
        if not os.path.isfile(self.klifs_full_catalog_tsv):
            return False
        try:
            df = pd.read_csv(self.klifs_full_catalog_tsv, sep="\t", nrows=2)
        except Exception:
            return False
        return not df.empty and "pdb" in df.columns and "chain" in df.columns

    def _count_klifs_chains_not_in_extract(self, inv: pd.DataFrame) -> int:
        """|KLIFS \\ InterPro extract| as distinct *(PDB, chain)*.

        Uses :attr:`klifs_full_catalog_tsv` when present (see :meth:`build_full_klifs_catalog`)
        so the Venn **KLIFS-only** region reflects the **full KLIFS API catalogue** of
        kinase structures. Otherwise falls back to the PDB-scoped ``structures_pdb_list``
        cache from :meth:`build_inventory`, then to ``klifs_venn_summary.json``.
        """
        op = self._our_pairs_from_inventory(inv)
        if self._has_nonempty_full_klifs_catalog():
            kdf = pd.read_csv(self.klifs_full_catalog_tsv, sep="\t")
            kp = self._pairs_from_klifs_pdb_response(kdf)
            return len(kp - op)
        if os.path.isfile(self.klifs_pdb_list_cache_tsv):
            kdf = pd.read_csv(self.klifs_pdb_list_cache_tsv, sep="\t")
            kp = self._pairs_from_klifs_pdb_response(kdf)
            return len(kp - op)
        if os.path.isfile(self.venn_summary_json):
            with open(self.venn_summary_json, encoding="utf-8") as fh:
                vs = json.load(fh)
            return int(
                vs.get("n_klifs_pairs_on_queried_pdbs_not_in_interpro_extract", 0)
            )
        return 0

    # ── Step 2: Venn diagram ─────────────────────────────────────────────
    def plot_venn(self) -> str:
        """Render an InterPro-vs-KLIFS Venn diagram and save to PNG.

        Region **01** is **KLIFS \\ InterPro extract** (distinct *PDB, chain*). If
        :meth:`build_full_klifs_catalog` has been run, that region uses the **full**
        KLIFS ``structures_list`` union across kinases; otherwise it uses only the
        ``structures_pdb_list`` results for PDB IDs in your extract.
        """
        import matplotlib.pyplot as plt

        df = pd.read_csv(self.inventory_tsv, sep="\t")
        n_total = len(df)
        n_klifs = int(df["in_klifs"].sum())
        n_only_inter = n_total - n_klifs

        use_full = self._has_nonempty_full_klifs_catalog()
        n_klifs_only = self._count_klifs_chains_not_in_extract(df)

        klifs_label = (
            "KLIFS\n(full API catalog)"
            if use_full
            else "KLIFS\n(PDB IDs in extract only)"
        )

        fig, ax = plt.subplots(figsize=(6.5, 6.2))
        try:
            from matplotlib_venn import venn2

            venn2(
                subsets={
                    "10": n_only_inter,
                    "01": n_klifs_only,
                    "11": n_klifs,
                },
                set_labels=(
                    "InterPro extract\n(chains in this study)",
                    klifs_label,
                ),
                ax=ax,
            )
        except ModuleNotFoundError:
            import matplotlib.patches as patches

            ax.add_patch(patches.Circle((0.40, 0.5), 0.36, color="tab:blue", alpha=0.35))
            ax.add_patch(patches.Circle((0.62, 0.5), 0.22, color="tab:orange", alpha=0.45))
            ax.text(0.25, 0.5, str(n_only_inter), ha="center", va="center", fontsize=14)
            ax.text(0.72, 0.72, str(n_klifs_only), ha="center", va="center", fontsize=12)
            ax.text(0.64, 0.5, str(n_klifs), ha="center", va="center",
                    fontsize=14, fontweight="bold")
            ax.text(0.28, 0.89, "InterPro", ha="center", va="center", fontsize=12)
            ax.text(0.74, 0.78, "KLIFS",  ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            if n_klifs_only:
                ax.text(
                    0.72,
                    0.62,
                    "KLIFS only\n(not in extract)",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            ax.text(
                0.5,
                0.06,
                "(install matplotlib-venn for proportional 3-region diagram)",
                ha="center",
                va="center",
                fontsize=9,
                color="0.35",
            )

        sub = (
            f"  ·  {n_klifs_only} KLIFS (PDB, chain) in the full API catalog are not in this extract"
            if use_full and n_klifs_only
            else (
                f"  ·  {n_klifs_only} KLIFS chains on your PDB IDs are not in this extract "
                "(run build_full_klifs_catalog() for the full-database count)"
                if n_klifs_only
                else ""
            )
        )
        fig.suptitle(
            f"InterPro extract vs KLIFS\n"
            f"{n_klifs} / {n_total} extract chains also in KLIFS"
            + sub,
            fontsize=11,
            y=0.98,
        )
        if use_full:
            foot = (
                "Region 01: KLIFS-only count = distinct (PDB, chain) from the union of "
                "/structures_list over all kinase_ID values in /kinase_information "
                "(HTTP 400 = no structures), minus this InterPro extract."
            )
        else:
            foot = (
                "Region 01 uses only KLIFS entries for PDB codes in your extract "
                "(structures_pdb_list). Run klifs.build_full_klifs_catalog() once "
                "(~1100 API calls, cached) for the full-database KLIFS-only crescent."
            )
        fig.text(
            0.5,
            0.01,
            foot,
            ha="center",
            fontsize=8.5,
            color="0.25",
        )
        fig.subplots_adjust(bottom=0.14, top=0.86)
        fig.savefig(self.venn_png, dpi=200, bbox_inches="tight")
        print(f"Saved Venn diagram → {self.venn_png}")
        plt.show()
        return self.venn_png

    # ── Structure-level (PDB ID) inventory + Venn ─────────────────────────
    @staticmethod
    def _list_downloaded_pdb_ids(pdb_dir: str) -> List[str]:
        """Return sorted unique PDB IDs from ``.pdb`` / ``.cif`` files in *pdb_dir*."""
        if not os.path.isdir(pdb_dir):
            raise FileNotFoundError(f"PDB directory not found: {pdb_dir}")
        ids: Set[str] = set()
        for name in os.listdir(pdb_dir):
            lower = name.lower()
            if not (lower.endswith(".pdb") or lower.endswith(".cif")):
                continue
            if not os.path.isfile(os.path.join(pdb_dir, name)):
                continue
            stem, _ = os.path.splitext(name)
            pid = stem.strip().upper()
            if pid:
                ids.add(pid)
        return sorted(ids)

    def _klifs_pdb_ids_from_full_catalog(self) -> Optional[Set[str]]:
        """Unique PDB IDs from the full KLIFS catalog TSV, or ``None`` if unavailable."""
        if not self._has_nonempty_full_klifs_catalog():
            return None
        kdf = pd.read_csv(self.klifs_full_catalog_tsv, sep="\t")
        return set(kdf["pdb"].astype(str).str.upper())

    def build_structure_inventory(
        self,
        pdb_dir: str = "Results/InterPro_PDBs",
        force: bool = False,
    ) -> pd.DataFrame:
        """Build a per-structure (PDB ID) inventory of KLIFS membership.

        Scans downloaded structure files in *pdb_dir* and marks each unique PDB ID
        as present in KLIFS or not. Prefer the full-catalog cache from
        :meth:`build_full_klifs_catalog`; otherwise query ``structures_pdb_list``.

        Writes ``klifs_structure_inventory.tsv`` and ``klifs_structure_venn_summary.json``.
        Re-uses the cached TSV when present (pass ``force=True`` to rebuild).
        """
        if (not force) and os.path.isfile(self.structure_inventory_tsv):
            print(f"Reusing cached structure inventory: {self.structure_inventory_tsv}")
            if not os.path.isfile(self.structure_venn_summary_json):
                print(
                    f"  Note: {self.structure_venn_summary_json} missing — "
                    "run build_structure_inventory(force=True) once to refresh summary."
                )
            return pd.read_csv(self.structure_inventory_tsv, sep="\t")

        pdb_ids = self._list_downloaded_pdb_ids(pdb_dir)
        print(
            f"Scanning {len(pdb_ids)} downloaded structures in {pdb_dir!r} against KLIFS"
        )

        klifs_pdbs = self._klifs_pdb_ids_from_full_catalog()
        used_full = klifs_pdbs is not None
        if klifs_pdbs is None:
            print(
                "  Full KLIFS catalog not found — querying structures_pdb_list "
                "for downloaded PDB IDs only. Run build_full_klifs_catalog() for "
                "the full-database KLIFS-only crescent."
            )
            klifs_df = self._klifs_structures_for_pdbs(pdb_ids)
            if klifs_df.empty:
                klifs_pdbs = set()
            else:
                klifs_pdbs = set(klifs_df["pdb"].astype(str).str.upper())

        rows = [
            {"pdb_id": pid, "in_klifs": pid in klifs_pdbs}
            for pid in pdb_ids
        ]
        df = pd.DataFrame(rows)
        df.to_csv(self.structure_inventory_tsv, sep="\t", index=False)

        interpro_set = set(pdb_ids)
        n_in_klifs = int(df["in_klifs"].sum())
        n_klifs_only = len(klifs_pdbs - interpro_set) if used_full else 0
        if not used_full:
            # PDB-scoped: KLIFS hits among queried IDs that somehow aren't in download
            # (normally 0); still record for completeness.
            n_klifs_only = len(klifs_pdbs - interpro_set)

        venn_summary = {
            "n_downloaded_structures": len(pdb_ids),
            "n_downloaded_in_klifs": n_in_klifs,
            "n_downloaded_not_in_klifs": len(df) - n_in_klifs,
            "n_klifs_structures_not_in_download": int(n_klifs_only),
            "used_full_klifs_catalog": used_full,
            "pdb_dir": pdb_dir,
            "explanation": (
                "n_klifs_structures_not_in_download counts distinct PDB IDs in the "
                "KLIFS full API catalog (or structures_pdb_list fallback) minus "
                "downloaded InterPro PDB IDs."
            ),
        }
        with open(self.structure_venn_summary_json, "w", encoding="utf-8") as fh:
            json.dump(venn_summary, fh, indent=2)
        print(f"  Structure Venn counts →  {self.structure_venn_summary_json}")
        print(
            f"  in KLIFS    : {n_in_klifs} / {len(df)} structures"
            f"  →  {self.structure_inventory_tsv}"
        )
        return df

    def plot_venn_structures(self) -> str:
        """Render an InterPro-vs-KLIFS Venn diagram at the **structure** (PDB ID) level.

        Region **01** is **KLIFS \\ downloaded InterPro PDBs**. If
        :meth:`build_full_klifs_catalog` has been run, that region uses unique PDB
        IDs from the full KLIFS catalogue; otherwise it falls back to
        ``structures_pdb_list`` hits for the downloaded IDs only.
        """
        import matplotlib.pyplot as plt

        if not os.path.isfile(self.structure_inventory_tsv):
            raise FileNotFoundError(
                f"Missing {self.structure_inventory_tsv}. "
                "Run build_structure_inventory() first."
            )

        df = pd.read_csv(self.structure_inventory_tsv, sep="\t")
        n_total = len(df)
        n_klifs = int(df["in_klifs"].sum())
        n_only_inter = n_total - n_klifs
        interpro_set = set(df["pdb_id"].astype(str).str.upper())

        use_full = self._has_nonempty_full_klifs_catalog()
        klifs_pdbs = self._klifs_pdb_ids_from_full_catalog()
        if klifs_pdbs is None:
            # Fall back to inventory membership only → KLIFS-only crescent is 0
            # unless summary was written with a prior full/partial query.
            if os.path.isfile(self.structure_venn_summary_json):
                with open(self.structure_venn_summary_json, encoding="utf-8") as fh:
                    vs = json.load(fh)
                n_klifs_only = int(vs.get("n_klifs_structures_not_in_download", 0))
            else:
                n_klifs_only = 0
        else:
            n_klifs_only = len(klifs_pdbs - interpro_set)

        klifs_label = (
            "KLIFS\n(full API catalog)"
            if use_full
            else "KLIFS\n(PDB IDs in download only)"
        )

        fig, ax = plt.subplots(figsize=(6.5, 6.2))
        try:
            from matplotlib_venn import venn2

            venn2(
                subsets={
                    "10": n_only_inter,
                    "01": n_klifs_only,
                    "11": n_klifs,
                },
                set_labels=(
                    "InterPro download\n(structures in this study)",
                    klifs_label,
                ),
                ax=ax,
            )
        except ModuleNotFoundError:
            import matplotlib.patches as patches

            ax.add_patch(patches.Circle((0.40, 0.5), 0.36, color="tab:blue", alpha=0.35))
            ax.add_patch(patches.Circle((0.62, 0.5), 0.22, color="tab:orange", alpha=0.45))
            ax.text(0.25, 0.5, str(n_only_inter), ha="center", va="center", fontsize=14)
            ax.text(0.72, 0.72, str(n_klifs_only), ha="center", va="center", fontsize=12)
            ax.text(0.64, 0.5, str(n_klifs), ha="center", va="center",
                    fontsize=14, fontweight="bold")
            ax.text(0.28, 0.89, "InterPro", ha="center", va="center", fontsize=12)
            ax.text(0.74, 0.78, "KLIFS", ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            if n_klifs_only:
                ax.text(
                    0.72,
                    0.62,
                    "KLIFS only\n(not in download)",
                    ha="center",
                    va="center",
                    fontsize=9,
                )
            ax.text(
                0.5,
                0.06,
                "(install matplotlib-venn for proportional 3-region diagram)",
                ha="center",
                va="center",
                fontsize=9,
                color="0.35",
            )

        sub = (
            f"  ·  {n_klifs_only} KLIFS structures in the full API catalog are not in this download"
            if use_full and n_klifs_only
            else (
                f"  ·  {n_klifs_only} KLIFS structures on your PDB IDs are not in this download "
                "(run build_full_klifs_catalog() for the full-database count)"
                if n_klifs_only
                else ""
            )
        )
        fig.suptitle(
            f"InterPro download vs KLIFS\n"
            f"{n_klifs} / {n_total} download structures also in KLIFS"
            + sub,
            fontsize=11,
            y=0.98,
        )
        if use_full:
            foot = (
                "Region 01: KLIFS-only count = distinct PDB IDs from the union of "
                "/structures_list over all kinase_ID values in /kinase_information "
                "(HTTP 400 = no structures), minus this InterPro download."
            )
        else:
            foot = (
                "Region 01 uses only KLIFS entries for PDB codes in your download "
                "(structures_pdb_list). Run klifs.build_full_klifs_catalog() once "
                "(~1100 API calls, cached) for the full-database KLIFS-only crescent."
            )
        fig.text(
            0.5,
            0.01,
            foot,
            ha="center",
            fontsize=8.5,
            color="0.25",
        )
        fig.subplots_adjust(bottom=0.14, top=0.86)
        fig.savefig(self.structure_venn_png, dpi=200, bbox_inches="tight")
        print(f"Saved structure Venn diagram → {self.structure_venn_png}")
        plt.show()
        return self.structure_venn_png

    # ── Step 3: DFG/APE per KLIFS chain + copy ───────────────────────────
    @staticmethod
    def _chain_sequence(pdb_path: str) -> Tuple[Optional[str], Optional[List[int]]]:
        """Return (sequence_str, residue_numbers) for the chain in *pdb_path*."""
        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()
        try:
            structure = parser.get_structure("x", pdb_path)
        except Exception:
            return None, None
        seq_parts: List[str] = []
        res_nums: List[int] = []
        for pp in ppb.build_peptides(structure):
            seq_parts.append(str(pp.get_sequence()))
            for residue in pp:
                res_nums.append(residue.id[1])
        seq = "".join(seq_parts)
        if not seq:
            return None, None
        return seq, res_nums

    @staticmethod
    def _resolve_match_residues_columns(mr: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort detection of KLIFS-position and PDB-residue-number columns."""
        pos_col = next(
            (c for c in mr.columns if c.lower() in {"klifs_position", "klifsposition", "position"}),
            None,
        )
        res_col = next(
            (c for c in mr.columns if c.lower() in {"xray_position", "pdb_position", "resnum", "residue", "pdb_resnum"}),
            None,
        )
        return pos_col, res_col

    @staticmethod
    def _dfg_row_from_match_residues(
        mr: pd.DataFrame, pos_col: str, klifs_pos_dfg: int
    ) -> pd.DataFrame:
        """Return the row(s) for pocket position *klifs_pos_dfg* (canonical DFG-Asp = 81).

        Current KLIFS responses use labels like ``xDFG.81``; older rows may use integer ``81``.
        """
        col = mr[pos_col]
        s = col.astype(str)
        tag = f"xDFG.{klifs_pos_dfg}"
        hit = mr.loc[s == tag]
        if not hit.empty:
            return hit
        hit = mr.loc[col == klifs_pos_dfg]
        if not hit.empty:
            return hit
        hit = mr.loc[s == str(klifs_pos_dfg)]
        if not hit.empty:
            return hit
        return mr.loc[s.str.endswith(f".{klifs_pos_dfg}")]

    def _build_small_molecule_index(self) -> Dict[str, List[str]]:
        """Map ``PDBID_CHAIN`` → list of paired small-molecule filenames."""
        index: Dict[str, List[str]] = defaultdict(list)
        if not os.path.isdir(self.small_molecule_dir):
            return {}
        for f in os.listdir(self.small_molecule_dir):
            if not f.lower().endswith(".pdb"):
                continue
            index[_chain_id_prefix(f)].append(f)
        return index

    @staticmethod
    def _basename_from_skip_entry(raw: str) -> str:
        """Recover ``PDB_CHAIN`` basename from a skip-log entry (may include ``(motif)``)."""
        s = (raw or "").strip()
        if "\t" in s:
            s = s.split("\t", 1)[0].strip()
        if "(" in s:
            s = s.split("(", 1)[0].strip()
        return s

    def _write_hmmer_input_basenames(self, skipped: Dict[str, List[str]]) -> int:
        """Write ``hmmer_input_basenames.txt`` for :class:`workflow.hmmer_diagnostics.HMMERDiagnostics`.

        Returns the number of distinct basenames written.
        """
        inv = pd.read_csv(self.inventory_tsv, sep="\t")
        bases: set = set(
            inv.loc[~inv["in_klifs"], "basename"].astype(str).str.strip()
        )
        for names in skipped.values():
            for raw in names:
                b = self._basename_from_skip_entry(raw)
                if b:
                    bases.add(b)
        os.makedirs(self.klifs_dir, exist_ok=True)
        path = self.hmmer_input_basenames_txt
        with open(path, "w", encoding="utf-8") as fh:
            for b in sorted(bases):
                fh.write(f"{b}\n")
        print(
            f"HMMER input basenames (not in KLIFS ∪ KLIFS filter skips) → {path} "
            f"({len(bases)} chains)"
        )
        return len(bases)

    def run_filter(
        self,
        klifs_pos_dfg: int = KLIFS_POS_DFG_ASP,
        apply_pseudokinase_exclusion: bool = True,
    ) -> Dict:
        """Locate DFG/APE for every KLIFS chain, copy passing PDBs + small molecules,
        and write the activation-loop TSV consumed by the Tukey filter.

        Always processes **every** row in the inventory with ``in_klifs`` true (after
        optional pseudokinase exclusion inside the loop). There is **no** parameter to
        limit the number of structures; partial runs are not supported by this API.
        """
        if not os.path.isfile(self.inventory_tsv):
            raise FileNotFoundError(
                f"Inventory missing: {self.inventory_tsv}. "
                f"Call .build_inventory() first."
            )

        df = pd.read_csv(self.inventory_tsv, sep="\t")
        klifs_rows = df[df["in_klifs"]].copy().reset_index(drop=True)

        excluded: set = set()
        if apply_pseudokinase_exclusion:
            excluded = load_excluded_pseudokinase_basenames(
                self.pseudo_basenames_path, required=True
            )
            print(f"Loaded {len(excluded)} pseudokinase basenames to exclude.")

        _clear_and_make_dir(self.target_protein_dir)
        _clear_and_make_dir(self.target_ligand_dir)

        sm_index = self._build_small_molecule_index()
        loop_records: List[dict] = []
        n_protein = n_ligand = 0
        skipped: Dict[str, List[str]] = defaultdict(list)

        for _, row in tqdm(
            klifs_rows.iterrows(),
            total=len(klifs_rows),
            desc="KLIFS DFG/APE",
        ):
            base = row["basename"]
            if base in excluded:
                skipped["pseudokinase"].append(base)
                continue

            sids = [int(s) for s in str(row["klifs_structure_ids"]).split(",") if s]
            if not sids:
                skipped["no_structure_id"].append(base)
                continue

            sid = sids[0]   # first hit; could rank by quality_score in future
            mr = self._klifs_match_residues(sid)
            if mr is None or mr.empty:
                skipped["no_match_residues"].append(base)
                continue
            pos_col, res_col = self._resolve_match_residues_columns(mr)
            if pos_col is None or res_col is None:
                skipped["unexpected_klifs_columns"].append(base)
                continue

            dfg_row = self._dfg_row_from_match_residues(mr, pos_col, klifs_pos_dfg)
            if dfg_row.empty:
                skipped["no_dfg_position"].append(base)
                continue
            try:
                dfg_pdb_resnum = int(dfg_row.iloc[0][res_col])
            except Exception:
                skipped["dfg_resnum_parse"].append(base)
                continue

            pdb_file = os.path.join(self.input_dir, base + ".pdb")
            if not os.path.isfile(pdb_file):
                skipped["missing_pdb"].append(base)
                continue

            seq, resnums = self._chain_sequence(pdb_file)
            if seq is None or resnums is None:
                skipped["seq_extraction_fail"].append(base)
                continue

            try:
                idx_dfg = resnums.index(dfg_pdb_resnum)
            except ValueError:
                skipped["dfg_resnum_not_in_chain"].append(base)
                continue

            if seq[idx_dfg:idx_dfg + 3] != "DFG":
                skipped["dfg_mismatch"].append(
                    f"{base}({seq[idx_dfg:idx_dfg + 3]})"
                )
                continue

            ape_idx = seq.find("APE", idx_dfg + 3)
            if ape_idx == -1:
                skipped["no_ape_downstream"].append(base)
                continue

            loop_seq = seq[idx_dfg + 3:ape_idx]
            loop_records.append({
                "pdb_basename": base,
                "loop_sequence": loop_seq,
                "dfg_residue_number": dfg_pdb_resnum,
                "ape_residue_number": resnums[ape_idx],
                "klifs_structure_id": sid,
            })

            shutil.copy2(pdb_file, self.target_protein_dir)
            n_protein += 1
            for fname in sm_index.get(base, []):
                shutil.copy2(
                    os.path.join(self.small_molecule_dir, fname),
                    self.target_ligand_dir,
                )
                n_ligand += 1

        loop_df = pd.DataFrame(loop_records)
        os.makedirs(os.path.dirname(self.loop_tsv) or ".", exist_ok=True)
        loop_df.to_csv(self.loop_tsv, sep="\t", index=False)

        print()
        print(f"Wrote {len(loop_df)} activation-loop sequences → {self.loop_tsv}")
        print(f"Copied {n_protein} chain PDBs            → {self.target_protein_dir}")
        print(f"Copied {n_ligand} small-molecule files   → {self.target_ligand_dir}")

        if skipped:
            print("\nSkipped chains (by reason):")
            for reason, names in skipped.items():
                preview = ", ".join(names[:5])
                tail = f" ... and {len(names) - 5} more" if len(names) > 5 else ""
                print(f"  {reason:<30s}: {len(names):>5d}  e.g. {preview}{tail}")
            with open(self.klifs_failures_txt, "w") as fh:
                for reason, names in skipped.items():
                    for n in names:
                        fh.write(f"{n}\t{reason}\n")
            print(f"\nSaved skip list → {self.klifs_failures_txt}")

        n_hmmer_input = self._write_hmmer_input_basenames(skipped)

        n_total_chains = sum(
            1
            for f in os.listdir(self.input_dir)
            if f.lower().endswith(".pdb")
        )
        summary = {
            "n_total_chains": n_total_chains,
            "n_klifs_inventory_rows": int(len(klifs_rows)),
            "n_klifs_passed_motif_filter": int(n_protein),
            "n_loop_records": int(len(loop_df)),
            "n_ligand_files_copied": int(n_ligand),
            "n_hmmer_input_basenames": int(n_hmmer_input),
            "hmmer_input_basenames_path": self.hmmer_input_basenames_txt,
            "skipped_counts": {k: len(v) for k, v in skipped.items()},
        }
        os.makedirs(self.klifs_dir, exist_ok=True)
        with open(self.filter_summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved KLIFS filter summary → {self.filter_summary_json}")

        return {
            "n_protein": n_protein,
            "n_ligand": n_ligand,
            "n_loop_records": len(loop_df),
            "loop_tsv": self.loop_tsv,
            "target_protein_dir": self.target_protein_dir,
            "target_ligand_dir": self.target_ligand_dir,
            "skipped": {k: list(v) for k, v in skipped.items()},
            "summary_path": self.filter_summary_json,
            "hmmer_input_basenames": self.hmmer_input_basenames_txt,
            "n_hmmer_input_basenames": n_hmmer_input,
        }

    # ── Match-residues cache (all 85 pocket positions per KLIFS chain) ───
    @property
    def match_residues_cache_tsv(self) -> str:
        return os.path.join(self.klifs_dir, "match_residues_cache.tsv")

    def build_match_residues_cache(
        self,
        force: bool = False,
        delay_s: float = 0.05,
    ) -> str:
        """Fetch all 85 KLIFS pocket-position mappings for every KLIFS chain.

        Writes ``Results/KLIFS/match_residues_cache.tsv`` with columns
        ``basename``, ``structure_id``, ``klifs_position``, ``xray_position``.
        Re-uses the cached file on subsequent calls unless *force* is ``True``.

        Parameters
        ----------
        force : bool
            Re-fetch even if the cache already exists.
        delay_s : float
            Seconds to sleep between API calls (default 0.05 ≈ 8 min total).

        Returns
        -------
        str
            Path to the written TSV.
        """
        if os.path.isfile(self.match_residues_cache_tsv) and not force:
            print(f"Reusing cached match_residues: {self.match_residues_cache_tsv}")
            return self.match_residues_cache_tsv

        if not os.path.isfile(self.inventory_tsv):
            raise FileNotFoundError(
                f"Inventory missing: {self.inventory_tsv}. "
                "Run build_inventory() first."
            )

        inv = pd.read_csv(self.inventory_tsv, sep="\t")
        klifs_rows = inv[inv["in_klifs"]].copy().reset_index(drop=True)
        print(f"Fetching match_residues for {len(klifs_rows)} KLIFS chains …")

        rows: List[dict] = []
        n_ok = n_fail = 0

        for _, row in tqdm(klifs_rows.iterrows(), total=len(klifs_rows),
                           desc="match_residues cache"):
            basename = str(row["basename"])
            sids_str = str(row.get("klifs_structure_ids", ""))
            sids = [int(s) for s in sids_str.split(",") if s.strip().isdigit()]
            if not sids:
                n_fail += 1
                continue
            sid = sids[0]
            mr = self._klifs_match_residues(sid)
            if mr is None or mr.empty:
                n_fail += 1
                if delay_s:
                    time.sleep(delay_s)
                continue
            pos_col, res_col = self._resolve_match_residues_columns(mr)
            if pos_col is None or res_col is None:
                n_fail += 1
                if delay_s:
                    time.sleep(delay_s)
                continue
            for _, mr_row in mr.iterrows():
                xray = mr_row[res_col]
                try:
                    xray_int: Optional[int] = int(xray)
                except (TypeError, ValueError):
                    xray_int = None  # covers NaN, '_', '', and other non-numeric sentinels
                rows.append({
                    "basename":       basename,
                    "structure_id":   sid,
                    "klifs_position": str(mr_row[pos_col]),
                    "xray_position":  xray_int,
                })
            n_ok += 1
            if delay_s:
                time.sleep(delay_s)

        df = pd.DataFrame(rows, columns=["basename", "structure_id",
                                          "klifs_position", "xray_position"])
        df.to_csv(self.match_residues_cache_tsv, sep="\t", index=False)
        print(
            f"Cached {n_ok}/{len(klifs_rows)} structures "
            f"({n_fail} with no data) → {self.match_residues_cache_tsv}"
        )
        return self.match_residues_cache_tsv

    # ── Supplement: download KLIFS-only chains missing from the extract ──
    @staticmethod
    def _rcsb_download_pdb(pdb_id: str, timeout: int = 60) -> Optional[str]:
        """Fetch PDB text from RCSB.  Returns the text or ``None`` on failure."""
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        try:
            r = requests.get(url, timeout=timeout)
        except Exception as exc:
            print(f"  Download failed ({pdb_id}): {exc}")
            return None
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {pdb_id}")
            return None
        return r.text

    @staticmethod
    def _extract_chain_biopython(
        pdb_text: str,
        pdb_id: str,
        chain_id: str,
        out_path: str,
    ) -> bool:
        """Write a single chain (ATOM records only) from *pdb_text* to *out_path*.

        Uses :class:`Bio.PDB.PDBParser` / ``PDBIO`` so the output matches the format
        of the existing InterPro-extracted chain PDBs.  Returns ``True`` on success.
        """
        from Bio.PDB import PDBIO, Select

        class _ChainSelect(Select):
            def accept_chain(self, chain):
                return chain.id == chain_id

            def accept_residue(self, residue):
                return residue.id[0] == " "  # ATOM only (no HETATM / waters)

        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))
        except Exception as exc:
            print(f"  Bio.PDB parse error ({pdb_id}): {exc}")
            return False

        model = structure[0]
        if chain_id not in [c.id for c in model.get_chains()]:
            print(f"  Chain {chain_id} not found in {pdb_id}")
            return False

        pdbio = PDBIO()
        pdbio.set_structure(model[chain_id])
        try:
            pdbio.save(out_path, _ChainSelect())
        except Exception as exc:
            print(f"  Write error ({pdb_id}_{chain_id}): {exc}")
            return False
        return True

    @staticmethod
    def _extract_small_molecules_biopython(
        pdb_text: str,
        pdb_id: str,
        chain_id: str,
        out_path: str,
        *,
        min_heavy_atoms: int = 6,
    ) -> bool:
        """Write non-water HETATM residues for *chain_id* to *out_path*.

        Mirrors what ``PDBChainExtractor`` does for the InterPro extract.
        """
        from Bio.PDB import PDBIO, Select

        _WATER = {"HOH", "WAT", "H2O", "DOD"}
        _STD_AA = {
            "ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
            "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL",
        }

        class _HetSelect(Select):
            def accept_chain(self, chain):
                return chain.id == chain_id

            def accept_residue(self, residue):
                het, _, _ = residue.id
                if het == " ":
                    return False
                rn = residue.resname.strip().upper()
                if rn in _WATER or rn in _STD_AA:
                    return False
                heavy = sum(1 for a in residue if a.element not in ("H", "D", ""))
                return heavy >= min_heavy_atoms

        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))
        except Exception:
            return False

        model = structure[0]
        if chain_id not in [c.id for c in model.get_chains()]:
            return False

        pdbio = PDBIO()
        pdbio.set_structure(model[chain_id])
        try:
            pdbio.save(out_path, _HetSelect())
        except Exception:
            return False

        # Remove zero-byte file (chain had no qualifying small molecules)
        if os.path.isfile(out_path) and os.path.getsize(out_path) == 0:
            os.remove(out_path)
            return False
        return True

    @staticmethod
    def _extract_chain_with_ligands_biopython(
        pdb_text: str,
        pdb_id: str,
        chain_id: str,
        out_path: str,
        *,
        min_heavy_atoms: int = 6,
    ) -> bool:
        """Write protein ATOM residues plus non-water HETATM ligands for *chain_id*.

        This is the ligand–chain complex counterpart to :meth:`_extract_chain_biopython`.
        """
        from Bio.PDB import PDBIO, Select

        _WATER = {"HOH", "WAT", "H2O", "DOD"}
        _STD_AA = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        }

        class _ComplexSelect(Select):
            def accept_chain(self, chain):
                return chain.id == chain_id

            def accept_residue(self, residue):
                het, _, _ = residue.id
                if het == " ":
                    return True  # standard protein residues
                rn = residue.resname.strip().upper()
                if rn in _WATER or rn in _STD_AA:
                    return False
                heavy = sum(1 for a in residue if a.element not in ("H", "D", ""))
                return heavy >= min_heavy_atoms

        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, io.StringIO(pdb_text))
        except Exception as exc:
            print(f"  Bio.PDB parse error ({pdb_id}): {exc}")
            return False

        model = structure[0]
        if chain_id not in [c.id for c in model.get_chains()]:
            print(f"  Chain {chain_id} not found in {pdb_id}")
            return False

        pdbio = PDBIO()
        pdbio.set_structure(model[chain_id])
        try:
            pdbio.save(out_path, _ComplexSelect())
        except Exception as exc:
            print(f"  Write error ({pdb_id}_{chain_id} complex): {exc}")
            return False
        return True

    def download_all_klifs_chains(
        self,
        protein_dir: str = "Results/KLIFS_protein_chains/",
        ligand_dir: str = "Results/KLIFS_small_molecules/",
        force: bool = False,
        rcsb_pdb_dir: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict:
        """Download **all** unique KLIFS ``(PDB, chain)`` pairs into dedicated dirs.

        Requires :meth:`build_full_klifs_catalog` (uses ``klifs_full_catalog_pdb_chain.tsv``).

        For each catalogue pair:

        1. Download the full PDB from RCSB (cached under *rcsb_pdb_dir*).
        2. Write protein-only chain (ATOM) → ``protein_dir / {PDB}_{CHAIN}.pdb``.
        3. Write ligand–chain complex (ATOM + qualifying HETATM) →
           ``ligand_dir / {PDB}_{CHAIN}.pdb``.

        Existing outputs are skipped unless *force* is ``True``.
        """
        if not os.path.isfile(self.klifs_full_catalog_tsv):
            raise FileNotFoundError(
                f"Full KLIFS catalog not found: {self.klifs_full_catalog_tsv}. "
                "Run klifs.build_full_klifs_catalog() first."
            )

        cat = pd.read_csv(self.klifs_full_catalog_tsv, sep="\t")
        pairs: List[Tuple[str, str]] = sorted({
            (str(row["pdb"]).upper(), str(row["chain"]).upper())
            for _, row in cat.iterrows()
            if pd.notna(row.get("pdb")) and pd.notna(row.get("chain"))
        })
        if not pairs:
            print("Full KLIFS catalog is empty — nothing to download.")
            return {
                "n_pairs": 0,
                "n_protein_added": 0,
                "n_complex_added": 0,
                "n_skipped_exists": 0,
                "n_download_fail": 0,
                "n_extract_fail": 0,
            }

        print(f"Downloading {len(pairs)} KLIFS (PDB, chain) pairs →")
        print(f"  protein:  {protein_dir}")
        print(f"  complexes: {ligand_dir}")

        cache_dir = rcsb_pdb_dir or os.path.join(self.klifs_dir, "rcsb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(protein_dir, exist_ok=True)
        os.makedirs(ligand_dir, exist_ok=True)

        n_prot = n_cplx = n_skipped = n_dl_fail = n_ex_fail = 0
        from itertools import groupby

        for pdb_id, group in tqdm(
            groupby(pairs, key=lambda x: x[0]),
            total=len({p for p, _ in pairs}),
            desc="KLIFS RCSB download+extract",
        ):
            chain_ids = [c for _, c in group]
            pdb_cache = os.path.join(cache_dir, f"{pdb_id}.pdb")

            need_any = force or any(
                (not os.path.isfile(os.path.join(protein_dir, f"{pdb_id}_{c}.pdb")))
                or (not os.path.isfile(os.path.join(ligand_dir, f"{pdb_id}_{c}.pdb")))
                for c in chain_ids
            )
            if not need_any:
                n_skipped += len(chain_ids)
                continue

            if not os.path.isfile(pdb_cache):
                pdb_text = self._rcsb_download_pdb(pdb_id, timeout=timeout)
                if pdb_text is None:
                    n_dl_fail += len(chain_ids)
                    continue
                with open(pdb_cache, "w", encoding="utf-8") as fh:
                    fh.write(pdb_text)
            else:
                with open(pdb_cache, encoding="utf-8") as fh:
                    pdb_text = fh.read()

            for chain_id in chain_ids:
                basename = f"{pdb_id}_{chain_id}"
                protein_out = os.path.join(protein_dir, basename + ".pdb")
                complex_out = os.path.join(ligand_dir, basename + ".pdb")

                prot_exists = os.path.isfile(protein_out)
                cplx_exists = os.path.isfile(complex_out)
                if prot_exists and cplx_exists and not force:
                    n_skipped += 1
                    continue

                if (not prot_exists) or force:
                    ok = self._extract_chain_biopython(
                        pdb_text, pdb_id, chain_id, protein_out
                    )
                    if not ok:
                        n_ex_fail += 1
                        continue
                    n_prot += 1

                if (not cplx_exists) or force:
                    ok_c = self._extract_chain_with_ligands_biopython(
                        pdb_text, pdb_id, chain_id, complex_out
                    )
                    if ok_c:
                        n_cplx += 1
                    # Complex extract can fail when Bio.PDB cannot write; protein
                    # may still have succeeded. Count extract fail only if protein
                    # was also missing this round and we already continued above.
                    elif not os.path.isfile(complex_out):
                        # Still count as complex miss but do not abort the pair
                        pass

        print(
            f"\nKLIFS full-catalog download summary:\n"
            f"  Catalogue pairs                 : {len(pairs)}\n"
            f"  Protein chains written          : {n_prot}\n"
            f"  Ligand–chain complexes written  : {n_cplx}\n"
            f"  Skipped (both files existed)    : {n_skipped}\n"
            f"  Failed — download error         : {n_dl_fail}\n"
            f"  Failed — protein extract error  : {n_ex_fail}"
        )
        return {
            "n_pairs": len(pairs),
            "n_protein_added": n_prot,
            "n_complex_added": n_cplx,
            "n_skipped_exists": n_skipped,
            "n_download_fail": n_dl_fail,
            "n_extract_fail": n_ex_fail,
        }

    def supplement_klifs_only_chains(
        self,
        force: bool = False,
        rcsb_pdb_dir: Optional[str] = None,
        timeout: int = 60,
    ) -> Dict:
        """Download and extract KLIFS (PDB, chain) pairs absent from the InterPro extract.

        Requires :meth:`build_full_klifs_catalog` and :meth:`build_inventory` to have
        been run first (both are cached to disk).

        For each missing pair the method:

        1. Downloads the full PDB from RCSB (cached to *rcsb_pdb_dir* when provided,
           otherwise to a temporary ``Results/KLIFS/rcsb_cache/`` directory).
        2. Extracts the protein chain (ATOM records only) →
           ``input_dir / {PDB}_{CHAIN}.pdb``.
        3. Extracts paired small molecules (HETATM, ≥6 heavy atoms, non-water) →
           ``small_molecule_dir / {PDB}_{CHAIN}.pdb``.
        4. Appends the new rows to ``klifs_chain_inventory.tsv`` (marked
           ``in_klifs=True``) and rewrites ``klifs_chain_basenames.txt``.

        After this call, run :meth:`run_filter` as usual — it will process the
        supplemented chains alongside the original ones.

        Parameters
        ----------
        force : bool
            Re-download and re-extract even if the chain PDB already exists.
        rcsb_pdb_dir : str, optional
            Directory to cache full PDB downloads.  Defaults to
            ``Results/KLIFS/rcsb_cache/``.
        timeout : int
            HTTP request timeout in seconds.

        Returns
        -------
        dict
            Summary with ``n_added``, ``n_skipped_exists``, ``n_download_fail``,
            ``n_extract_fail``, ``new_basenames``.
        """
        if not os.path.isfile(self.klifs_full_catalog_tsv):
            raise FileNotFoundError(
                f"Full KLIFS catalog not found: {self.klifs_full_catalog_tsv}. "
                "Run klifs.build_full_klifs_catalog() first."
            )
        if not os.path.isfile(self.inventory_tsv):
            raise FileNotFoundError(
                f"Inventory not found: {self.inventory_tsv}. "
                "Run klifs.build_inventory() first."
            )

        cat = pd.read_csv(self.klifs_full_catalog_tsv, sep="\t")
        inv = pd.read_csv(self.inventory_tsv, sep="\t")

        cat_pairs: Set[Tuple[str, str]] = set(
            zip(cat["pdb"].str.upper(), cat["chain"].str.upper())
        )
        inv_pairs: Set[Tuple[str, str]] = set(
            zip(inv["pdb_id"].str.upper(), inv["chain_id"].str.upper())
        )
        missing: List[Tuple[str, str]] = sorted(cat_pairs - inv_pairs)

        if not missing:
            print("No KLIFS-only chains to supplement — inventory is already complete.")
            return {
                "n_added": 0, "n_skipped_exists": 0,
                "n_download_fail": 0, "n_extract_fail": 0,
                "new_basenames": [],
            }

        print(f"Found {len(missing)} KLIFS-only (PDB, chain) pairs to supplement.")

        cache_dir = rcsb_pdb_dir or os.path.join(self.klifs_dir, "rcsb_cache")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.small_molecule_dir, exist_ok=True)

        # Build structure_ID lookup from catalog for inventory update
        sid_lookup: Dict[Tuple[str, str], int] = {}
        if "structure_ID" in cat.columns:
            for _, row in cat.iterrows():
                key = (str(row["pdb"]).upper(), str(row["chain"]).upper())
                sid_lookup[key] = int(row["structure_ID"]) if pd.notna(row.get("structure_ID")) else -1

        n_added = n_skipped = n_dl_fail = n_ex_fail = 0
        new_rows: List[dict] = []
        new_basenames: List[str] = []

        # Group by PDB to avoid re-downloading the same file for multiple chains
        from itertools import groupby
        for pdb_id, pairs in groupby(missing, key=lambda x: x[0]):
            chain_ids = [c for _, c in pairs]
            pdb_cache = os.path.join(cache_dir, f"{pdb_id}.pdb")

            # Download once per PDB
            if not os.path.isfile(pdb_cache):
                pdb_text = self._rcsb_download_pdb(pdb_id, timeout=timeout)
                if pdb_text is None:
                    n_dl_fail += len(chain_ids)
                    continue
                with open(pdb_cache, "w", encoding="utf-8") as fh:
                    fh.write(pdb_text)
            else:
                with open(pdb_cache, encoding="utf-8") as fh:
                    pdb_text = fh.read()

            for chain_id in chain_ids:
                basename = f"{pdb_id}_{chain_id}"
                protein_out = os.path.join(self.input_dir, basename + ".pdb")

                if os.path.isfile(protein_out) and not force:
                    n_skipped += 1
                    # Still register in new_rows so inventory is updated
                    sid = sid_lookup.get((pdb_id, chain_id), -1)
                    new_rows.append({
                        "basename": basename,
                        "pdb_id": pdb_id,
                        "chain_id": chain_id,
                        "in_klifs": True,
                        "klifs_structure_ids": str(sid) if sid >= 0 else "",
                    })
                    new_basenames.append(basename)
                    continue

                ok = self._extract_chain_biopython(pdb_text, pdb_id, chain_id, protein_out)
                if not ok:
                    n_ex_fail += 1
                    continue

                # Small molecules (best-effort; missing file is not an error)
                sm_out = os.path.join(self.small_molecule_dir, basename + ".pdb")
                self._extract_small_molecules_biopython(pdb_text, pdb_id, chain_id, sm_out)

                sid = sid_lookup.get((pdb_id, chain_id), -1)
                new_rows.append({
                    "basename": basename,
                    "pdb_id": pdb_id,
                    "chain_id": chain_id,
                    "in_klifs": True,
                    "klifs_structure_ids": str(sid) if sid >= 0 else "",
                })
                new_basenames.append(basename)
                n_added += 1

        # Update inventory on disk
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            # Avoid duplicates if supplement is called twice
            existing_bases = set(inv["basename"].astype(str))
            new_df = new_df[~new_df["basename"].isin(existing_bases)]
            if not new_df.empty:
                updated = pd.concat([inv, new_df], ignore_index=True)
                updated.to_csv(self.inventory_tsv, sep="\t", index=False)
                updated.loc[updated["in_klifs"], "basename"].to_csv(
                    self.klifs_basenames_txt, index=False, header=False
                )
                print(
                    f"Updated inventory: +{len(new_df)} rows → {self.inventory_tsv}"
                )

        print(
            f"\nSupplement summary:\n"
            f"  Added (extracted + registered) : {n_added}\n"
            f"  Skipped (file already existed) : {n_skipped}\n"
            f"  Failed — download error        : {n_dl_fail}\n"
            f"  Failed — extraction error      : {n_ex_fail}\n"
            f"  Total new basenames registered : {len(new_basenames)}"
        )
        return {
            "n_added": n_added,
            "n_skipped_exists": n_skipped,
            "n_download_fail": n_dl_fail,
            "n_extract_fail": n_ex_fail,
            "new_basenames": new_basenames,
        }


def plot_alignment_source_histogram(
    chain_input_dir: str = DEFAULT_INPUT_DIR,
    klifs_summary_json: str = "Results/KLIFS/klifs_filter_summary.json",
    hmmer_summary_json: str = "Results/HMMER/hmmer_alignment_summary.json",
    output_png: str = "Results/KLIFS/alignment_source_histogram.png",
    *,
    display: bool = True,
) -> Dict[str, object]:
    """Three-bar histogram: KLIFS motif success, Pfam ``hmmscan`` hit (non-KLIFS), left out.

    * **KLIFS** — chains that passed the KLIFS DFG/APE filter
      (``n_klifs_passed_motif_filter`` in ``klifs_filter_summary.json``).
    * **HMMER** — chains in ``hmmer_input_basenames.txt`` (after ``run_filter``: not in
      KLIFS **or** KLIFS motif filter failed) with at least one Pfam kinase ``hmmscan``
      hit (``n_pfam_kinase_hmmscan_hit`` in ``hmmer_alignment_summary.json``).
    * **Left out** — all other extracted chain PDBs (KLIFS failures, no Pfam hit,
      missing files, etc.).

    Run after §2.1.2 and §2.1.3 so both JSON summaries exist; missing files count as 0.
    """
    n_total = sum(
        1 for f in os.listdir(chain_input_dir) if f.lower().endswith(".pdb")
    )

    n_klifs = 0
    if os.path.isfile(klifs_summary_json):
        with open(klifs_summary_json, encoding="utf-8") as fh:
            n_klifs = int(json.load(fh).get("n_klifs_passed_motif_filter", 0))

    n_hmmer = 0
    if os.path.isfile(hmmer_summary_json):
        with open(hmmer_summary_json, encoding="utf-8") as fh:
            n_hmmer = int(json.load(fh).get("n_pfam_kinase_hmmscan_hit", 0))

    n_left = max(0, n_total - n_klifs - n_hmmer)

    import matplotlib.pyplot as plt

    labels = ("KLIFS\n(motif OK)", "HMMER\n(Pfam hit)", "Left out")
    counts = (n_klifs, n_hmmer, n_left)
    colors = ("#2ca02c", "#ff7f0e", "#7f7f7f")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_ylabel("Number of structures")
    ax.set_title(
        "Activation-loop mapping source\n"
        f"Total extracted chains: {n_total}"
    )
    for bar, n in zip(bars, counts):
        if n > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(int(n)),
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=200)
    print(f"Saved histogram → {output_png}")
    if display:
        plt.show()
    else:
        plt.close(fig)

    return {
        "n_total": n_total,
        "n_klifs_motif_ok": n_klifs,
        "n_hmmer_pfam_hit": n_hmmer,
        "n_left_out": n_left,
        "output_png": output_png,
    }


_KLIFS_SKIP_LABELS: Dict[str, str] = {
    "pseudokinase": "Pseudokinase excluded",
    "no_structure_id": "No KLIFS structure ID",
    "no_match_residues": "No match_residues response",
    "unexpected_klifs_columns": "Unexpected KLIFS columns",
    "no_dfg_position": "No DFG position in KLIFS",
    "dfg_resnum_parse": "DFG residue parse error",
    "missing_pdb": "Missing PDB file",
    "seq_extraction_fail": "Sequence extraction failed",
    "dfg_resnum_not_in_chain": "DFG residue not in chain",
    "dfg_mismatch": "DFG triplet mismatch",
    "no_ape_downstream": "No APE downstream of DFG",
}

_KLIFS_SKIP_COLORS: Dict[str, str] = {
    "pseudokinase": "#bee5eb",
    "dfg_mismatch": "#f5c6cb",
    "no_ape_downstream": "#ffeeba",
    "no_dfg_position": "#fde2e4",
    "dfg_resnum_parse": "#fde2e4",
    "dfg_resnum_not_in_chain": "#fde2e4",
    "missing_pdb": "#d6d8d9",
    "seq_extraction_fail": "#d6d8d9",
    "no_match_residues": "#d6d8d9",
    "unexpected_klifs_columns": "#d6d8d9",
    "no_structure_id": "#d6d8d9",
}


def _klifs_skip_label(reason: str) -> str:
    return _KLIFS_SKIP_LABELS.get(reason, reason.replace("_", " ").capitalize())


def plot_klifs_filter_skip_histogram(
    skipped: Optional[Dict[str, List[str]]] = None,
    *,
    n_passed: Optional[int] = None,
    summary_json: str = "Results/KLIFS/klifs_filter_summary.json",
    output_png: str = "Results/KLIFS/klifs_filter_skip_histogram.png",
    display: bool = True,
) -> Dict[str, object]:
    """Horizontal bar chart of KLIFS DFG/APE filter outcomes: passed + per-flag skips.

    Data can be supplied in-memory from :meth:`KLIFSOverlap.run_filter` (``skipped``
    dict) or reloaded from ``klifs_filter_summary.json`` (``skipped_counts``).
    """
    summary: Dict[str, object] = {}
    if os.path.isfile(summary_json):
        with open(summary_json, encoding="utf-8") as fh:
            summary = json.load(fh)

    if skipped is not None:
        skip_counts = {k: len(v) for k, v in skipped.items()}
    else:
        skip_counts = dict(summary.get("skipped_counts") or {})

    if n_passed is None:
        n_passed = int(summary.get("n_klifs_passed_motif_filter", 0))

    n_inventory = int(summary.get("n_klifs_inventory_rows", 0))
    if not n_inventory and skip_counts:
        n_inventory = int(n_passed) + sum(skip_counts.values())

    skip_sorted = sorted(skip_counts.items(), key=lambda kv: kv[1], reverse=True)

    import matplotlib.pyplot as plt

    y_labels = ["Passed (DFG/APE OK)"] + [
        _klifs_skip_label(reason) for reason, _ in skip_sorted
    ]
    counts = [int(n_passed)] + [int(n) for _, n in skip_sorted]
    colors = ["#2ca02c"] + [
        _KLIFS_SKIP_COLORS.get(reason, "#d6d8d9") for reason, _ in skip_sorted
    ]

    fig_h = max(4.0, 0.45 * len(y_labels) + 1.5)
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    bars = ax.barh(
        y_labels,
        counts,
        color=colors,
        edgecolor="white",
        linewidth=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Number of KLIFS chains")
    title = "KLIFS DFG/APE filter outcomes"
    if n_inventory:
        title += f"\nKLIFS inventory rows: {n_inventory}"
    ax.set_title(title)

    xmax = max(counts) if counts else 1
    for bar, n in zip(bars, counts):
        if n > 0:
            ax.text(
                bar.get_width() + max(xmax * 0.01, 5),
                bar.get_y() + bar.get_height() / 2,
                str(int(n)),
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    fig.savefig(output_png, dpi=200)
    print(f"Saved histogram → {output_png}")
    if display:
        plt.show()
    else:
        plt.close(fig)

    return {
        "n_passed": int(n_passed),
        "n_klifs_inventory_rows": n_inventory,
        "skipped_counts": skip_counts,
        "output_png": output_png,
    }
