"""Pfam-HMM diagnostics for kinase chains **not in KLIFS**.

**Diagnostics only** — this module does *not* filter, copy or modify any
PDB files. It computes a panel of per-chain metrics so the user can design
the actual filter logic in a follow-up step.

Chain list (FASTA inputs) — by default ``Results/KLIFS/non_klifs_chain_basenames.txt``
(chains absent from the KLIFS database, written by
:meth:`workflow.klifs_filter.KLIFSOverlap.build_inventory`).
Pass an explicit path to ``non_klifs_list`` to override.

Metrics produced (one row per non-KLIFS chain) include:

* ``best_evalue``, ``best_bitscore``
* ``hmm_from``, ``hmm_to``, ``hmm_model_len`` (envelope on the HMM)
* ``env_from``, ``env_to``, ``env_qlen`` (envelope on the query / chain)
* ``model_coverage`` = ``(hmm_to - hmm_from + 1) / hmm_model_len``
* ``n_significant_hits`` (multi-hit / fusion detection)
* ``chain_length`` (sequence length parsed from the PDB)
* ``missing_ca_fraction`` over the HMM-mapped query span
* ``interpro_span_start``, ``interpro_span_end``, ``interpro_span_length``
* ``span_overlap_fraction`` and ``span_length_ratio`` for the HMM envelope vs
  the InterPro domain span (check #4)
* Boolean flags: ``flag_low_bitscore``, ``flag_low_model_coverage``,
  ``flag_short_chain``, ``flag_high_missing_ca``,
  ``flag_span_mismatch``, ``flag_multi_hit``
* ``any_flag`` — true if any of the above flags fired

The list of chains with ``any_flag == True`` is written to
``Results/HMMER/hmmer_failures.txt``.

The Pfam Protein-kinase HMM is auto-downloaded from InterPro
(``https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00069?annotation=hmm``)
to ``Results/HMMER/Pkinase.hmm`` if not already on disk.

External requirements
---------------------
* ``hmmer`` on ``PATH`` — provides ``hmmpress``, ``hmmscan``.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from tqdm.auto import tqdm

from workflow.chain_basenames import load_excluded_pseudokinase_basenames

try:
    import requests
except ImportError as exc:
    raise ImportError(
        "The 'requests' package is required. Install with: pip install requests"
    ) from exc

DEFAULT_HMMER_DIR = "Results/HMMER"
DEFAULT_HMM_PATH = "Results/HMMER/Pkinase.hmm"
DEFAULT_INTERPRO_SPANS_TSV = "Results/HMMER/interpro_domain_spans.tsv"

KLIFS_NON_KLIFS_FALLBACK = "Results/KLIFS/non_klifs_chain_basenames.txt"
DEFAULT_PSEUDO_FILE = "Results/excluded_pseudokinase_basenames.txt"

PFAM_PKINASE_ACCESSION = "PF00069"

# Schema for ``fetch_interpro_spans`` / merge in ``compile_diagnostics``.
_INTERPRO_SPAN_COLS = ["pdb_id", "chain_id", "span_start", "span_end"]


def _looks_like_gzip(data: bytes) -> bool:
    return len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B


def _is_valid_hmmer3_hmm_file(path: str) -> bool:
    """Return True if *path* looks like a plain-text HMMER3 profile (not gzip / garbage)."""
    try:
        with open(path, "rb") as fh:
            buf = fh.read(120)
    except OSError:
        return False
    if len(buf) < 8:
        return False
    if _looks_like_gzip(buf):
        return False
    return buf.lstrip().startswith((b"HMMER3", b"# HMMER3"))


def _strip_gzip_if_needed(data: bytes) -> bytes:
    """Decompress *data* when it is gzip-wrapped (InterPro ``application/gzip``)."""
    if _looks_like_gzip(data):
        return gzip.decompress(data)
    return data

# Where the Pfam Protein-kinase HMM can be fetched from.
PFAM_HMM_URLS = [
    # InterPro JSON wrapper around the Pfam HMM.
    f"https://www.ebi.ac.uk/interpro/api/entry/pfam/{PFAM_PKINASE_ACCESSION}?annotation=hmm",
    # Direct Pfam/FTP route (gzipped HMM).
    f"https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz",
]

_CHAIN_LONG = re.compile(r"^([A-Za-z0-9]{4})_chain([A-Za-z0-9]+)$")
_CHAIN_SHORT = re.compile(r"^([A-Za-z0-9]{4})_([A-Za-z0-9]+)$")

# Heuristic thresholds.  Tune later; failures here only build a *candidate*
# rejection list — the user designs the actual filter on top.
DEFAULT_THRESHOLDS = {
    "min_bitscore":         50.0,
    "min_model_coverage":   0.70,
    "min_chain_length":     150,
    "max_missing_ca_frac":  0.20,
    "max_span_ratio_diff":  0.40,    # |HMM envelope length - InterPro span length| / max(...)
}


def _parse_basename(basename: str) -> Optional[Tuple[str, str]]:
    m = _CHAIN_LONG.match(basename) or _CHAIN_SHORT.match(basename)
    if not m:
        return None
    return m.group(1).upper(), m.group(2)


class HMMERDiagnostics:
    """Run hmmscan on chains outside the successful KLIFS motif filter."""

    def __init__(
        self,
        chain_dir: str = "Results/InterPro_protein_chains/",
        non_klifs_list: Optional[str] = None,
        hmmer_dir: str = DEFAULT_HMMER_DIR,
        hmm_path: str = DEFAULT_HMM_PATH,
        interpro_spans_tsv: str = DEFAULT_INTERPRO_SPANS_TSV,
        interpro_entry_id: str = "IPR011009",
        thresholds: Optional[Dict[str, float]] = None,
        hmmscan_binary: str = "hmmscan",
        hmmpress_binary: str = "hmmpress",
        pseudo_basenames_path: str = DEFAULT_PSEUDO_FILE,
        apply_pseudokinase_exclusion: bool = True,
        target_chains_dir: str = "Results/motif_filtered_chains/",
        loop_tsv: str = "Results/activation_loop_sequences.tsv",
    ):
        self.chain_dir = chain_dir
        self._chain_basename_list_override = non_klifs_list
        self.hmmer_dir = hmmer_dir
        self.hmm_path = hmm_path
        self.interpro_spans_tsv = interpro_spans_tsv
        self.interpro_entry_id = interpro_entry_id
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.hmmscan_binary = hmmscan_binary
        self.hmmpress_binary = hmmpress_binary
        self.pseudo_basenames_path = pseudo_basenames_path
        self.apply_pseudokinase_exclusion = apply_pseudokinase_exclusion
        self.target_chains_dir = target_chains_dir
        self.loop_tsv = loop_tsv
        self._seqs: Dict[str, str] = {}
        self._resnums: Dict[str, List[int]] = {}
        os.makedirs(self.hmmer_dir, exist_ok=True)

    @property
    def chain_basename_list_path(self) -> str:
        """Path to the newline-separated basename list consumed by ``build_non_klifs_fasta``.

        By default this is ``Results/KLIFS/non_klifs_chain_basenames.txt`` — chains that
        are **not** in KLIFS at all.  Pass ``non_klifs_list`` to ``__init__`` to override.
        """
        if self._chain_basename_list_override is not None:
            return self._chain_basename_list_override
        return KLIFS_NON_KLIFS_FALLBACK

    # ── output paths ─────────────────────────────────────────────────────
    @property
    def fasta_path(self) -> str:
        return os.path.join(self.hmmer_dir, "non_klifs_sequences.fasta")

    @property
    def domtblout_path(self) -> str:
        return os.path.join(self.hmmer_dir, "hmmscan_domtblout.tsv")

    @property
    def diagnostics_tsv(self) -> str:
        return os.path.join(self.hmmer_dir, "hmmer_diagnostics.tsv")

    @property
    def failures_txt(self) -> str:
        return os.path.join(self.hmmer_dir, "hmmer_failures.txt")

    @property
    def alignment_summary_json(self) -> str:
        return os.path.join(self.hmmer_dir, "hmmer_alignment_summary.json")

    # ── Step 1: ensure HMM is on disk and pressed ────────────────────────
    def ensure_hmm(self) -> str:
        """Download and ``hmmpress`` the Pfam Protein-kinase HMM if missing."""
        self._purge_corrupt_hmm_if_needed()

        if (
            os.path.isfile(self.hmm_path)
            and all(
                os.path.isfile(self.hmm_path + ext)
                for ext in (".h3i", ".h3p", ".h3f", ".h3m")
            )
        ):
            return self.hmm_path

        os.makedirs(os.path.dirname(self.hmm_path) or ".", exist_ok=True)
        if not os.path.isfile(self.hmm_path):
            print(f"Downloading Pfam {PFAM_PKINASE_ACCESSION} HMM …")
            for url in PFAM_HMM_URLS:
                try:
                    r = requests.get(url, timeout=120)
                except Exception as exc:
                    print(f"  failed: {url} ({exc})")
                    continue
                if r.status_code != 200:
                    print(f"  HTTP {r.status_code} for {url}")
                    continue
                content = r.content
                if "annotation=hmm" in url or "annotation%3Dhmm" in url:
                    # InterPro serves ``application/gzip`` (profile text, gzip-compressed).
                    content = _strip_gzip_if_needed(content)
                    self._write_bytes(self.hmm_path, content)
                else:
                    # Pfam-A.hmm.gz — extract just the Pkinase HMM block.
                    try:
                        text = gzip.decompress(content).decode()
                    except Exception:
                        text = content.decode(errors="ignore")
                    block = self._extract_hmm_block(text, PFAM_PKINASE_ACCESSION)
                    if not block:
                        continue
                    self._write_bytes(self.hmm_path, block.encode())
                break
            if not os.path.isfile(self.hmm_path):
                raise RuntimeError(
                    "Could not download Pfam Pkinase HMM. "
                    "Place it manually at: " + self.hmm_path
                )
            print(f"HMM saved → {self.hmm_path}")

        # hmmpress for fast hmmscan
        print(f"Running hmmpress on {self.hmm_path}")
        try:
            subprocess.run(
                [self.hmmpress_binary, "-f", self.hmm_path],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not find `hmmpress`. Install HMMER (e.g. `conda install -c bioconda hmmer`)."
            ) from exc
        except subprocess.CalledProcessError as exc:
            tail = (exc.stderr or "")[-800:] if exc.stderr else ""
            raise RuntimeError(
                f"hmmpress failed (exit {exc.returncode}). "
                f"If {self.hmm_path} is corrupt, delete it and the .h3* siblings, then retry.\n"
                f"stderr (tail):\n{tail}"
            ) from exc
        return self.hmm_path

    def _purge_corrupt_hmm_if_needed(self) -> None:
        """Remove gzip-as-.hmm or other invalid profiles so we can re-download."""
        if not os.path.isfile(self.hmm_path):
            return
        if _is_valid_hmmer3_hmm_file(self.hmm_path):
            return
        print(f"Removing invalid HMM file (not HMMER3 text): {self.hmm_path}")
        for ext in ("", ".h3i", ".h3p", ".h3f", ".h3m"):
            p = self.hmm_path + ext if ext else self.hmm_path
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

    @staticmethod
    def _write_bytes(path: str, data: bytes) -> None:
        with open(path, "wb") as fh:
            fh.write(data)

    @staticmethod
    def _extract_hmm_block(text: str, accession: str) -> Optional[str]:
        """Pull a single HMM block (HMMER profile) out of a multi-HMM file."""
        out_lines: List[str] = []
        buf: List[str] = []
        for line in text.splitlines(keepends=True):
            buf.append(line)
            if line.startswith("//"):
                block = "".join(buf)
                if f"ACC   {accession}" in block:
                    return block
                buf = []
        return None

    # ── Step 2: extract sequences for non-KLIFS chains ──────────────────
    def build_non_klifs_fasta(self) -> Tuple[str, Dict[str, str], Dict[str, List[int]]]:
        """Build a FASTA of non-KLIFS chain sequences; also return resnum maps."""
        if not os.path.isfile(self.chain_basename_list_path):
            raise FileNotFoundError(
                f"Missing basename list: {self.chain_basename_list_path}. "
                f"Run ``KLIFSOverlap.build_inventory()`` and either ``run_filter()`` "
                f"(writes merged {KLIFS_HMMER_INPUT_MERGED}) or ensure "
                f"{KLIFS_NON_KLIFS_FALLBACK} exists."
            )
        with open(self.chain_basename_list_path) as fh:
            basenames = [ln.strip() for ln in fh if ln.strip()]

        excluded: set = set()
        if self.apply_pseudokinase_exclusion:
            excluded = load_excluded_pseudokinase_basenames(
                self.pseudo_basenames_path, required=True
            )
            n_before = len(basenames)
            basenames = [b for b in basenames if b not in excluded]
            print(
                f"Pseudokinase exclusion: removed {n_before - len(basenames)} chains "
                f"({self.pseudo_basenames_path})"
            )

        print(f"Extracting sequences for {len(basenames)} chains ({self.chain_basename_list_path})")

        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()
        seqs: Dict[str, str] = {}
        resnums: Dict[str, List[int]] = {}
        for base in tqdm(basenames, desc="Read chains"):
            pdb_file = os.path.join(self.chain_dir, base + ".pdb")
            if not os.path.isfile(pdb_file):
                continue
            try:
                structure = parser.get_structure(base, pdb_file)
            except Exception:
                continue
            seq_parts: List[str] = []
            rn: List[int] = []
            for pp in ppb.build_peptides(structure):
                seq_parts.append(str(pp.get_sequence()))
                for r in pp:
                    rn.append(r.id[1])
            seq = "".join(seq_parts)
            if not seq:
                continue
            seqs[base] = seq
            resnums[base] = rn

        with open(self.fasta_path, "w") as fh:
            for name, seq in seqs.items():
                fh.write(f">{name}\n{seq}\n")
        print(f"FASTA: {len(seqs)} sequences → {self.fasta_path}")
        return self.fasta_path, seqs, resnums

    # ── Step 3: run hmmscan ──────────────────────────────────────────────
    def run_hmmscan(self) -> str:
        """Run ``hmmscan --domtblout`` (uses --cut_ga if the HMM is gathered).

        Writes the domtblout to :attr:`domtblout_path` and returns its path.
        """
        cmd = [
            self.hmmscan_binary,
            "--domtblout", self.domtblout_path,
            "--cpu", str(max(1, (os.cpu_count() or 2) - 1)),
            self.hmm_path,
            self.fasta_path,
        ]
        print(f"Running: {' '.join(cmd)}")
        try:
            r = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Could not find `hmmscan`. Install HMMER (e.g. "
                "`conda install -c bioconda hmmer`)."
            ) from exc
        if r.returncode != 0:
            print(r.stdout[-500:])
            print(r.stderr[-500:])
            raise RuntimeError(f"hmmscan failed with exit code {r.returncode}")
        print(f"domtblout → {self.domtblout_path}")
        return self.domtblout_path

    # ── Step 4: parse domtblout ──────────────────────────────────────────
    @staticmethod
    def parse_domtblout(path: str) -> pd.DataFrame:
        """Parse hmmscan --domtblout into a tidy DataFrame.

        Columns kept: target_name, target_acc, qlen, query_name, full_evalue,
        full_score, hmm_from, hmm_to, hmm_len (target length, M),
        env_from, env_to.
        """
        rows = []
        with open(path) as fh:
            for line in fh:
                if not line or line.startswith("#"):
                    continue
                cols = line.split()
                if len(cols) < 23:
                    continue
                rows.append({
                    "target_name":  cols[0],
                    "target_acc":   cols[1],
                    "hmm_len":      int(cols[2]),
                    "query_name":   cols[3],
                    "query_acc":    cols[4],
                    "qlen":         int(cols[5]),
                    "full_evalue":  float(cols[6]),
                    "full_score":   float(cols[7]),
                    "full_bias":    float(cols[8]),
                    "this_evalue":  float(cols[12]),
                    "this_score":   float(cols[13]),
                    "hmm_from":     int(cols[15]),
                    "hmm_to":       int(cols[16]),
                    "ali_from":     int(cols[17]),
                    "ali_to":       int(cols[18]),
                    "env_from":     int(cols[19]),
                    "env_to":       int(cols[20]),
                    "acc":          float(cols[21]),
                })
        df = pd.DataFrame(rows)
        return df

    # ── Step 5: InterPro span per chain ──────────────────────────────────
    def fetch_interpro_spans(
        self,
        pdb_chain_pairs: List[Tuple[str, str]],
        cache: bool = True,
    ) -> pd.DataFrame:
        """Query the InterPro API for the entry's coordinates on each (PDB, chain).

        Cached to :attr:`interpro_spans_tsv`. Returns a DataFrame with columns
        ``pdb_id``, ``chain_id``, ``span_start``, ``span_end``.
        """
        if cache and os.path.isfile(self.interpro_spans_tsv):
            print(f"Reusing cached InterPro spans: {self.interpro_spans_tsv}")
            try:
                df = pd.read_csv(self.interpro_spans_tsv, sep="\t")
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                return pd.DataFrame(columns=_INTERPRO_SPAN_COLS)
            if df.empty:
                return pd.DataFrame(columns=_INTERPRO_SPAN_COLS)
            for c in _INTERPRO_SPAN_COLS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[_INTERPRO_SPAN_COLS]

        out_rows: List[dict] = []
        for pdb_id, chain_id in tqdm(pdb_chain_pairs, desc="InterPro spans"):
            url = (
                f"https://www.ebi.ac.uk/interpro/api/entry/InterPro/"
                f"{self.interpro_entry_id}/structure/pdb/{pdb_id.lower()}"
            )
            try:
                r = requests.get(url, timeout=30)
            except Exception:
                continue
            if r.status_code != 200:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            results = data.get("results") or []
            for entry in results:
                for struct in entry.get("structures", []):
                    if struct.get("chain", "").lower() != chain_id.lower():
                        continue
                    for frag in struct.get("entry_protein_locations", [{}])[0].get("fragments", []):
                        out_rows.append({
                            "pdb_id":     pdb_id,
                            "chain_id":   chain_id,
                            "span_start": int(frag.get("start", -1)),
                            "span_end":   int(frag.get("end", -1)),
                        })
        df = (
            pd.DataFrame(out_rows, columns=_INTERPRO_SPAN_COLS)
            if out_rows
            else pd.DataFrame(columns=_INTERPRO_SPAN_COLS)
        )
        df.to_csv(self.interpro_spans_tsv, sep="\t", index=False)
        print(f"Saved {len(df)} InterPro spans → {self.interpro_spans_tsv}")
        return df

    # ── Step 6: missing-CA fraction in HMM-mapped span ───────────────────
    @staticmethod
    def missing_ca_fraction(
        pdb_path: str, env_from: int, env_to: int
    ) -> Optional[float]:
        """Fraction of expected residue positions without a CA atom in the
        env_from–env_to window (1-indexed, inclusive, residue ordinals along
        the chain peptide as enumerated by PPBuilder)."""
        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()
        try:
            structure = parser.get_structure("x", pdb_path)
        except Exception:
            return None
        idx = 0
        n_expected = 0
        n_missing = 0
        for pp in ppb.build_peptides(structure):
            for residue in pp:
                idx += 1
                if env_from <= idx <= env_to:
                    n_expected += 1
                    if "CA" not in residue:
                        n_missing += 1
        if n_expected == 0:
            return None
        return n_missing / n_expected

    def _write_hmmer_alignment_summary(
        self,
        *,
        n_pfam_kinase_hit: int,
        n_no_pfam_hit: int,
        n_non_klifs_input: int,
    ) -> None:
        """Write ``hmmer_alignment_summary.json`` for :func:`plot_alignment_source_histogram`."""
        summary = {
            "n_non_klifs_chains_input": int(n_non_klifs_input),
            "n_pfam_kinase_hmmscan_hit": int(n_pfam_kinase_hit),
            "n_no_pfam_kinase_hit": int(n_no_pfam_hit),
        }
        os.makedirs(self.hmmer_dir, exist_ok=True)
        with open(self.alignment_summary_json, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"Saved HMMER alignment summary → {self.alignment_summary_json}")

    # ── Step 7: compile diagnostics ──────────────────────────────────────
    def compile_diagnostics(self, copy_passing: bool = True) -> pd.DataFrame:
        """Combine domtblout + InterPro spans + missing-CA into a final TSV.

        When *copy_passing* is ``True`` (default), chains with no diagnostic flags
        are copied to ``target_chains_dir`` and their DFG→APE activation-loop
        sequences are appended to ``loop_tsv`` (same schema as the KLIFS path).
        """
        if not os.path.isfile(self.domtblout_path):
            raise FileNotFoundError(
                f"Missing domtblout: {self.domtblout_path}. Run .run_hmmscan() first."
            )
        dom = self.parse_domtblout(self.domtblout_path)
        all_non_klifs: set = set()
        if os.path.isfile(self.chain_basename_list_path):
            with open(self.chain_basename_list_path) as fh:
                all_non_klifs = {ln.strip() for ln in fh if ln.strip()}
        n_non_klifs_input = len(all_non_klifs)

        if dom.empty:
            print("WARNING: hmmscan returned no hits.")
            self._write_hmmer_alignment_summary(
                n_pfam_kinase_hit=0,
                n_no_pfam_hit=n_non_klifs_input,
                n_non_klifs_input=n_non_klifs_input,
            )
            return dom

        # Per-query: best hit (lowest E-value) + multi-hit count
        dom_sorted = dom.sort_values(["query_name", "full_evalue"])
        best = dom_sorted.groupby("query_name", as_index=False).first()
        sig = dom[dom["this_evalue"] < 1e-5]
        hit_counts = sig.groupby("query_name").size().rename("n_significant_hits")

        per_chain = best.merge(hit_counts, on="query_name", how="left")
        per_chain["n_significant_hits"] = per_chain["n_significant_hits"].fillna(0).astype(int)
        per_chain["model_coverage"] = (
            (per_chain["hmm_to"] - per_chain["hmm_from"] + 1) / per_chain["hmm_len"]
        )
        per_chain["env_qlen"] = per_chain["env_to"] - per_chain["env_from"] + 1
        per_chain["chain_length"] = per_chain["qlen"]

        # Parse (pdb_id, chain_id) from basename so we can join InterPro spans.
        per_chain["basename"] = per_chain["query_name"]
        parsed = per_chain["basename"].map(_parse_basename)
        per_chain["pdb_id"] = parsed.map(lambda x: x[0] if x else None)
        per_chain["chain_id"] = parsed.map(lambda x: x[1] if x else None)

        # InterPro spans
        pairs = list(zip(per_chain["pdb_id"].dropna(), per_chain["chain_id"].dropna()))
        if pairs:
            spans = self.fetch_interpro_spans(pairs)
            per_chain = per_chain.merge(
                spans, how="left", on=["pdb_id", "chain_id"]
            )
            per_chain["interpro_span_length"] = (
                per_chain["span_end"] - per_chain["span_start"] + 1
            )
        else:
            per_chain["span_start"] = pd.NA
            per_chain["span_end"] = pd.NA
            per_chain["interpro_span_length"] = pd.NA

        # Span comparison ratio
        denom = per_chain[["env_qlen", "interpro_span_length"]].max(axis=1)
        diff = (per_chain["env_qlen"] - per_chain["interpro_span_length"]).abs()
        per_chain["span_length_ratio_diff"] = (diff / denom).astype(float)

        # Missing-CA fraction
        miss = []
        for _, row in tqdm(
            per_chain.iterrows(), total=len(per_chain), desc="Missing-CA scan"
        ):
            pdb_file = os.path.join(self.chain_dir, row["basename"] + ".pdb")
            if not os.path.isfile(pdb_file):
                miss.append(None)
                continue
            miss.append(
                self.missing_ca_fraction(pdb_file, int(row["env_from"]), int(row["env_to"]))
            )
        per_chain["missing_ca_fraction"] = miss

        # Flags
        t = self.thresholds
        per_chain["flag_low_bitscore"] = per_chain["full_score"] < t["min_bitscore"]
        per_chain["flag_low_model_coverage"] = per_chain["model_coverage"] < t["min_model_coverage"]
        per_chain["flag_short_chain"] = per_chain["chain_length"] < t["min_chain_length"]
        per_chain["flag_high_missing_ca"] = (
            per_chain["missing_ca_fraction"].fillna(0) > t["max_missing_ca_frac"]
        )
        per_chain["flag_span_mismatch"] = (
            per_chain["span_length_ratio_diff"].fillna(0) > t["max_span_ratio_diff"]
        )
        per_chain["flag_multi_hit"] = per_chain["n_significant_hits"] > 1
        flag_cols = [c for c in per_chain.columns if c.startswith("flag_")]
        per_chain["any_flag"] = per_chain[flag_cols].any(axis=1)

        # Output
        cols = [
            "basename", "pdb_id", "chain_id",
            "chain_length", "qlen", "env_from", "env_to", "env_qlen",
            "hmm_from", "hmm_to", "hmm_len", "model_coverage",
            "full_evalue", "full_score", "n_significant_hits",
            "span_start", "span_end", "interpro_span_length",
            "span_length_ratio_diff", "missing_ca_fraction",
            *flag_cols, "any_flag",
        ]
        cols = [c for c in cols if c in per_chain.columns]
        per_chain[cols].to_csv(self.diagnostics_tsv, sep="\t", index=False)
        print(f"\nDiagnostics → {self.diagnostics_tsv}")

        failures = per_chain.loc[per_chain["any_flag"], "basename"]
        failures.to_csv(self.failures_txt, index=False, header=False)
        print(f"Flagged {len(failures)} chains as potential failures → {self.failures_txt}")

        # Plus the chains that had no hit at all (more severe than any flag).
        no_hit = sorted(all_non_klifs - set(per_chain["basename"]))
        if no_hit:
            with open(self.failures_txt, "a") as fh:
                for n in no_hit:
                    fh.write(f"{n}\n")
            print(f"  + {len(no_hit)} chains with NO Pfam-kinase hit (appended to {self.failures_txt})")

        self._write_hmmer_alignment_summary(
            n_pfam_kinase_hit=len(per_chain),
            n_no_pfam_hit=len(no_hit),
            n_non_klifs_input=n_non_klifs_input,
        )

        if copy_passing:
            self._copy_passing_and_extend_loops(per_chain[cols])

        return per_chain[cols]

    # ── Step 8: extract DFG/APE loop from HMM envelope (same logic as KLIFS) ──
    @staticmethod
    def _extract_dfg_ape_loop(
        seq: str,
        resnums: List[int],
        env_from: int,
        env_to: int,
    ) -> Optional[Tuple[str, Optional[int], Optional[int]]]:
        """Locate DFG and APE within the hmmscan envelope region of *seq*.

        *env_from* / *env_to* are 1-indexed query positions (as written by hmmscan).
        Mirrors the KLIFS path: DFG is found first within the envelope; APE is the
        first occurrence downstream of DFG in the full sequence.

        Returns ``(loop_sequence, dfg_resnum, ape_resnum)`` or ``None`` if DFG or APE
        cannot be located.
        """
        subseq = seq[env_from - 1:env_to]
        dfg_in_sub = subseq.find("DFG")
        if dfg_in_sub == -1:
            return None
        global_dfg = (env_from - 1) + dfg_in_sub
        ape_idx = seq.find("APE", global_dfg + 3)
        if ape_idx == -1:
            return None
        loop_seq = seq[global_dfg + 3:ape_idx]
        dfg_resnum = resnums[global_dfg] if resnums and global_dfg < len(resnums) else None
        ape_resnum = resnums[ape_idx] if resnums and ape_idx < len(resnums) else None
        return loop_seq, dfg_resnum, ape_resnum

    # ── Step 9: copy passing chains + append loop sequences ─────────────
    def _copy_passing_and_extend_loops(self, per_chain: pd.DataFrame) -> None:
        """Copy PDBs that passed all diagnostic flags to ``target_chains_dir`` and
        append their activation-loop sequences to ``loop_tsv``.

        Uses ``self._seqs`` / ``self._resnums`` populated by ``build_non_klifs_fasta``.
        Called automatically by ``compile_diagnostics`` when ``copy_passing=True``.
        """
        passing = per_chain[~per_chain["any_flag"]].copy()
        if passing.empty:
            print("No passing HMMER chains to copy.")
            return

        os.makedirs(self.target_chains_dir, exist_ok=True)
        n_copied = 0
        loop_records = []
        n_no_loop = 0

        for _, row in passing.iterrows():
            base = row["basename"]
            src = os.path.join(self.chain_dir, base + ".pdb")
            if os.path.isfile(src):
                shutil.copy2(src, self.target_chains_dir)
                n_copied += 1

            seq = self._seqs.get(base)
            rn = self._resnums.get(base, [])
            if seq is None:
                n_no_loop += 1
                continue
            result = self._extract_dfg_ape_loop(
                seq, rn, int(row["env_from"]), int(row["env_to"])
            )
            if result is None:
                n_no_loop += 1
                continue
            loop_seq, dfg_resnum, ape_resnum = result
            loop_records.append({
                "pdb_basename": base,
                "loop_sequence": loop_seq,
                "dfg_residue_number": dfg_resnum,
                "ape_residue_number": ape_resnum,
                "klifs_structure_id": None,
            })

        print(
            f"Copied {n_copied} passing HMMER chain PDBs → {self.target_chains_dir}\n"
            f"  DFG/APE loop found: {len(loop_records)} | not found: {n_no_loop}"
        )

        if not loop_records:
            return

        new_df = pd.DataFrame(loop_records)
        file_exists = os.path.isfile(self.loop_tsv)
        if file_exists:
            try:
                existing = pd.read_csv(self.loop_tsv, sep="\t")
                already = set(existing["pdb_basename"].astype(str))
                new_df = new_df[~new_df["pdb_basename"].isin(already)]
            except Exception:
                pass
        if new_df.empty:
            print("  All loop records already present in TSV — nothing appended.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(self.loop_tsv)), exist_ok=True)
        new_df.to_csv(
            self.loop_tsv, sep="\t", index=False,
            mode="a", header=not file_exists,
        )
        print(f"  Appended {len(new_df)} loop records → {self.loop_tsv}")

    # ── orchestration ────────────────────────────────────────────────────
    def run(self, copy_passing: bool = True) -> pd.DataFrame:
        """Convenience: ensure HMM, build FASTA, run hmmscan, compile diagnostics.

        When *copy_passing* is ``True`` (default), chains that pass all diagnostic
        flags are copied to ``target_chains_dir`` and their activation-loop sequences
        (DFG → APE, same logic as the KLIFS path) are appended to ``loop_tsv``.
        """
        self.ensure_hmm()
        _, self._seqs, self._resnums = self.build_non_klifs_fasta()
        self.run_hmmscan()
        return self.compile_diagnostics(copy_passing=copy_passing)
