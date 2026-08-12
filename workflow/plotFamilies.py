"""
plotFamilies.py
===============
Maps PDB structures (chain-split or whole-entry InterPro downloads) to
Manning kinome names and produces:
  - A CORAL (Kinome Render) CSV ready for upload
  - A failed_mappings.txt log
  - Vertical %-bar histograms for kinome group and species

Use ``run_interpro_pipeline()`` for ``Results/InterPro_PDBs``-style
directories of whole-entry ``XXXX.pdb`` files.
"""

import csv
import io
import json
import logging
import os
import re
import time
from collections import Counter
from math import log2
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Colorblind-friendly palette (Okabe-Ito / Wong) ───────────────

GROUP_COLORS = {
    "TK":       "#0072B2",
    "TKL":      "#E69F00",
    "STE":      "#56B4E9",
    "CMGC":     "#009E73",
    "AGC":      "#D55E00",
    "CAMK":     "#CC79A7",
    "CK1":      "#F0E442",
    "RGC":      "#999999",
    "Other":    "#882255",
    "Atypical": "#332288",
}

# ── API endpoints ─────────────────────────────────────────────────

RCSB_GRAPHQL_URL = "https://data.rcsb.org/graphql"

KINHUB_CSV_URL = (
    "https://raw.githubusercontent.com/openkinome/kinodata/"
    "master/data/KinHubKinaseList.csv"
)

RCSB_ENTRIES_QUERY = """
query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_polymer_entity_container_identifiers {
        auth_asym_ids
        uniprot_ids
      }
      rcsb_entity_source_organism {
        ncbi_scientific_name
        ncbi_taxonomy_id
      }
      uniprots {
        rcsb_uniprot_entry_name
      }
    }
  }
}
"""

# ═════════════════════════════════════════════════════════════════
#  1. Manning kinase classification
# ═════════════════════════════════════════════════════════════════

def load_manning_classification(cache_dir=None):
    """Load the KinHub kinase list (Manning classification).

    Downloads from GitHub on first call and caches as
    ``kinhub_kinases.csv`` next to this script (or in *cache_dir*).

    Returns a dict with four lookup indices:
        by_uniprot, by_hgnc, by_manning, by_xname
    each mapping an uppercase key → entry dict.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent
    cache_file = Path(cache_dir) / "kinhub_kinases.csv"

    csv_text = None
    if cache_file.exists():
        csv_text = cache_file.read_text()
        logger.info("Manning classification loaded from cache")
    else:
        try:
            resp = requests.get(KINHUB_CSV_URL, timeout=20)
            resp.raise_for_status()
            csv_text = resp.text
            cache_file.write_text(csv_text)
            logger.info("Downloaded & cached Manning classification")
        except Exception as exc:
            raise RuntimeError(
                f"Cannot fetch KinHub data ({exc}). Download manually from "
                f"{KINHUB_CSV_URL} and save as {cache_file}"
            ) from exc

    return _parse_kinhub_csv(csv_text)


def _parse_kinhub_csv(csv_text):
    by_uniprot, by_hgnc, by_manning, by_xname = {}, {}, {}, {}

    csv_text = csv_text.replace("\u00a0", " ")  # non-breaking → regular space
    for row in csv.DictReader(io.StringIO(csv_text)):
        entry = {
            "manning_name": row["Manning Name"].strip(),
            "hgnc_name":    (row.get("HGNC Name") or "").strip(),
            "xname":        (row.get("xName") or "").strip(),
            "group":        row["Group"].strip(),
            "family":       (row.get("Family") or "").strip(),
            "subfamily":    (row.get("SubFamily") or "").strip(),
            "uniprot_id":   (row.get("UniprotID") or "").strip(),
        }
        if entry["manning_name"].startswith("Domain2_"):
            continue

        uid = entry["uniprot_id"]
        if uid:
            by_uniprot.setdefault(uid, entry)
        for key, dct in [
            (entry["hgnc_name"].upper(), by_hgnc),
            (entry["manning_name"].upper(), by_manning),
            (entry["xname"].upper(), by_xname),
        ]:
            if key:
                dct.setdefault(key, entry)

    return {
        "by_uniprot": by_uniprot,
        "by_hgnc":    by_hgnc,
        "by_manning": by_manning,
        "by_xname":   by_xname,
    }

# ═════════════════════════════════════════════════════════════════
#  2. Filename parsing
# ═════════════════════════════════════════════════════════════════

_FNAME_RE = re.compile(
    r"^([A-Za-z0-9]{4})_([A-Za-z0-9]+?)(?:_reconstructed)?\.pdb$"
)
_PDB_ONLY_RE = re.compile(r"^([0-9][A-Za-z0-9]{3})\.pdb$", re.IGNORECASE)


def parse_pdb_filenames(directory):
    """Return sorted list of ``(PDB_ID, chain_ID)`` from *directory*."""
    pairs = []
    for fname in sorted(os.listdir(directory)):
        if fname == "combined.pdb" or not fname.endswith(".pdb"):
            continue
        m = _FNAME_RE.match(fname)
        if m:
            pairs.append((m.group(1).upper(), m.group(2)))
        else:
            logger.warning("Unparseable filename: %s", fname)
    return pairs


def collect_pdb_ids_from_dir(directory):
    """Return sorted unique PDB IDs from whole-entry or chain-split files.

    Accepts ``XXXX.pdb`` (InterPro downloads) and
    ``XXXX_Chain[_reconstructed].pdb`` (fitted MDA style).
    """
    ids = set()
    for fname in os.listdir(directory):
        if fname == "combined.pdb" or not fname.lower().endswith(".pdb"):
            continue
        m_chain = _FNAME_RE.match(fname)
        if m_chain:
            ids.add(m_chain.group(1).upper())
            continue
        m_only = _PDB_ONLY_RE.match(fname)
        if m_only:
            ids.add(m_only.group(1).upper())
            continue
        logger.warning("Unparseable filename: %s", fname)
    return sorted(ids)

# ═════════════════════════════════════════════════════════════════
#  3. RCSB PDB GraphQL batch mapping
# ═════════════════════════════════════════════════════════════════

def batch_pdb_to_uniprot(pdb_chain_pairs, batch_size=50):
    """Map (PDB, chain) → UniProt via RCSB ``entries`` GraphQL query.

    Batches by *unique PDB ID* (≈ 1 955 unique out of 2 523 files).

    Returns
    -------
    results : dict[(pdb, chain)] → info-dict
    failed  : list[(pdb, chain, reason)]
    """
    pdb_to_chains: dict[str, set | None] = {}
    for pdb, chain in pdb_chain_pairs:
        pdb_to_chains.setdefault(pdb, set()).add(chain)

    return _batch_rcsb_entries(pdb_to_chains, batch_size=batch_size)


def batch_pdb_ids_to_uniprot(pdb_ids, batch_size=50):
    """Map whole-entry PDB IDs → all polymer chains with UniProt via RCSB.

    Unlike ``batch_pdb_to_uniprot``, chains are discovered from RCSB rather
    than from filenames (for InterPro ``XXXX.pdb`` downloads).

    Returns
    -------
    results : dict[(pdb, chain)] → info-dict
    failed  : list[(pdb, chain | '', reason)]
    """
    pdb_to_chains = {pid.upper(): None for pid in pdb_ids}
    return _batch_rcsb_entries(pdb_to_chains, batch_size=batch_size)


def _batch_rcsb_entries(pdb_to_chains, batch_size=50):
    """Shared RCSB GraphQL batch mapper.

    *pdb_to_chains* maps PDB ID → set of wanted chains, or ``None`` to
    accept every polymer-entity chain that has UniProt IDs.
    """
    unique_pdbs = sorted(pdb_to_chains)
    total_batches = (len(unique_pdbs) + batch_size - 1) // batch_size
    results: dict = {}
    failed: list = []

    for start in range(0, len(unique_pdbs), batch_size):
        batch = unique_pdbs[start : start + batch_size]
        bnum = start // batch_size + 1

        try:
            resp = requests.post(
                RCSB_GRAPHQL_URL,
                json={"query": RCSB_ENTRIES_QUERY,
                      "variables": {"ids": batch}},
                timeout=60,
            )
            resp.raise_for_status()
            payload = resp.json()

            entries = (payload.get("data") or {}).get("entries") or []
            seen_pdbs: set[str] = set()

            for ent in entries:
                if ent is None:
                    continue
                pid = ent["rcsb_id"].upper()
                seen_pdbs.add(pid)
                wanted = pdb_to_chains.get(pid)
                matched: set[str] = set()

                for poly in ent.get("polymer_entities") or []:
                    ids = (
                        poly.get(
                            "rcsb_polymer_entity_container_identifiers"
                        )
                        or {}
                    )
                    auth_chs = ids.get("auth_asym_ids") or []
                    up_ids = ids.get("uniprot_ids") or []

                    orgs = poly.get("rcsb_entity_source_organism") or []
                    tax = orgs[0].get("ncbi_taxonomy_id") if orgs else None
                    org = (
                        orgs[0].get("ncbi_scientific_name", "")
                        if orgs
                        else ""
                    )

                    enames: list[str] = []
                    for u in poly.get("uniprots") or []:
                        enames.extend(
                            u.get("rcsb_uniprot_entry_name") or []
                        )

                    for ch in auth_chs:
                        if wanted is not None and ch not in wanted:
                            continue
                        matched.add(ch)
                        if up_ids:
                            results[(pid, ch)] = {
                                "uniprot_ids": up_ids,
                                "taxonomy_id": tax,
                                "organism": org,
                                "uniprot_entry_names": enames,
                            }
                        else:
                            failed.append(
                                (pid, ch, "No UniProt mapping in RCSB")
                            )

                if wanted is not None:
                    for ch in wanted - matched:
                        if (pid, ch) not in results:
                            failed.append(
                                (pid, ch, "Chain not in any polymer entity")
                            )
                elif not matched:
                    failed.append(
                        (pid, "", "No polymer chains with UniProt in RCSB")
                    )

            for pid in batch:
                if pid not in seen_pdbs:
                    wanted = pdb_to_chains.get(pid)
                    if wanted is None:
                        failed.append(
                            (pid, "", "PDB entry not found in RCSB")
                        )
                    else:
                        for ch in wanted:
                            failed.append(
                                (pid, ch, "PDB entry not found in RCSB")
                            )

        except Exception as exc:
            logger.error("RCSB batch %d error: %s", bnum, exc)
            for pid in batch:
                wanted = pdb_to_chains.get(pid)
                if wanted is None:
                    failed.append((pid, "", f"API error: {exc}"))
                else:
                    for ch in wanted:
                        failed.append((pid, ch, f"API error: {exc}"))

        if bnum % 5 == 0 or bnum == total_batches:
            logger.info(
                "  RCSB batch %d/%d – %d mapped so far",
                bnum, total_batches, len(results),
            )
        if start + batch_size < len(unique_pdbs):
            time.sleep(0.25)

    return results, failed

# ═════════════════════════════════════════════════════════════════
#  4. Manning-name resolution  (human + non-human ortholog)
# ═════════════════════════════════════════════════════════════════

def resolve_to_manning(rcsb_results, manning):
    """Resolve RCSB UniProt hits to Manning kinase names.

    Resolution order per entry:
      1. Direct UniProt-ID match  (covers all human kinases in KinHub)
      2. UniProt entry-name mnemonic match  (e.g. BRAF from BRAF_MOUSE)
      3. Batch UniProt gene-name lookup     (remaining non-human entries)
    """
    resolved, failed = [], []
    nonhuman_queue = []

    for (pdb, chain), info in rcsb_results.items():
        uid = info["uniprot_ids"][0]
        tax = info.get("taxonomy_id")
        enames = info.get("uniprot_entry_names", [])
        organism = info.get("organism", "")

        hit = manning["by_uniprot"].get(uid)

        if not hit:
            for en in enames:
                parts = en.split("_")
                if len(parts) >= 2:
                    mnem = parts[0].upper()
                    hit = (
                        manning["by_hgnc"].get(mnem)
                        or manning["by_manning"].get(mnem)
                        or manning["by_xname"].get(mnem)
                    )
                    if hit:
                        break

        if hit:
            resolved.append(_row(pdb, chain, hit, uid, organism=organism))
        elif tax != 9606:
            nonhuman_queue.append((pdb, chain, info))
        else:
            failed.append((
                pdb, chain,
                f"Human protein not in Manning kinase list "
                f"(UniProt: {uid})",
            ))

    if nonhuman_queue:
        nh_ok, nh_fail = _resolve_nonhuman(nonhuman_queue, manning)
        resolved.extend(nh_ok)
        failed.extend(nh_fail)

    return resolved, failed


def _row(pdb, chain, hit, uid, organism=""):
    return {
        "pdb": pdb, "chain": chain,
        "manning_name": hit["manning_name"],
        "hgnc_name":    hit["hgnc_name"],
        "group":        hit["group"],
        "family":       hit["family"],
        "uniprot_id":   uid or "",
        "organism":     organism or "",
    }


def _resolve_nonhuman(entries, manning):
    """Batch-resolve non-human kinases through UniProt gene names."""
    resolved, failed = [], []

    uid_map: dict[str, list] = {}
    for pdb, chain, info in entries:
        uid = info["uniprot_ids"][0]
        uid_map.setdefault(uid, []).append((pdb, chain, info))

    logger.info(
        "  Resolving %d non-human UniProt IDs via gene-name lookup …",
        len(uid_map),
    )
    gene_lookup = _batch_uniprot_genes(list(uid_map))

    for uid, group in uid_map.items():
        hit = None
        genes = gene_lookup.get(uid, [])
        for g in genes:
            gu = g.upper()
            hit = (
                manning["by_hgnc"].get(gu)
                or manning["by_manning"].get(gu)
                or manning["by_xname"].get(gu)
            )
            if hit:
                break

        for pdb, chain, info in group:
            if hit:
                resolved.append(
                    _row(
                        pdb, chain, hit, uid,
                        organism=info.get("organism", ""),
                    )
                )
            else:
                org = info.get("organism", "unknown")
                failed.append((
                    pdb, chain,
                    f"Non-human ({org}) – no Manning match "
                    f"(genes={genes}, UniProt={uid})",
                ))

    return resolved, failed


def _batch_uniprot_genes(uids, batch_size=200):
    """Fetch primary gene names from UniProt REST for a list of accessions."""
    result: dict[str, list[str]] = {}
    total = len(uids)
    for i in range(0, total, batch_size):
        batch = uids[i : i + batch_size]
        query = " OR ".join(f"accession:{u}" for u in batch)
        try:
            resp = requests.get(
                "https://rest.uniprot.org/uniprotkb/search",
                params={
                    "query": query,
                    "fields": "accession,gene_primary,gene_names",
                    "format": "json",
                    "size": str(len(batch)),
                },
                timeout=30,
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            for entry in resp.json().get("results", []):
                acc = entry.get("primaryAccession", "")
                names = []
                for g in entry.get("genes", []):
                    v = (g.get("geneName") or {}).get("value")
                    if v:
                        names.append(v)
                    for syn in g.get("synonyms", []):
                        sv = syn.get("value")
                        if sv:
                            names.append(sv)
                if names:
                    result[acc] = names
        except Exception as exc:
            logger.error("UniProt batch error: %s", exc)

        if i + batch_size < total:
            time.sleep(0.5)
    return result

# ═════════════════════════════════════════════════════════════════
#  5. Output generation
# ═════════════════════════════════════════════════════════════════

def generate_coral_csv(resolved, output_path):
    """Write a CORAL-ready CSV (deduplicated by Manning name).

    ``node_radius`` is scaled logarithmically by PDB structure count.
    """
    counts = Counter(e["manning_name"] for e in resolved)
    info: dict = {}
    for e in resolved:
        info.setdefault(e["manning_name"], e)

    rows = []
    for name in sorted(info):
        group = info[name]["group"]
        color = GROUP_COLORS.get(group, "#888888")
        n = counts[name]
        radius = min(max(round(log2(n + 1) + 3, 1), 3), 15)
        rows.append({
            "node": name,
            "node_color": color,
            "node_radius": radius,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info("CORAL CSV written: %s  (%d unique kinases)", output_path, len(df))
    return df


def log_failed_mappings(failed, output_path):
    """Write a human-readable failed-mappings log."""
    with open(output_path, "w") as fh:
        fh.write(f"{'PDB_ID':<8}{'Chain':<8}Reason\n")
        fh.write("-" * 90 + "\n")
        for pdb, chain, reason in sorted(failed):
            fh.write(f"{pdb:<8}{chain:<8}{reason}\n")
    logger.info("Failed mappings written: %s  (%d entries)", output_path, len(failed))


def plot_family_histogram(resolved, output_path=None):
    """Colorblind-friendly bar chart of kinase-group distribution.

    Counts *unique* Manning names per group.
    """
    seen: dict[str, str] = {}
    for e in resolved:
        seen.setdefault(e["manning_name"], e["group"])

    gcounts = Counter(seen.values())
    groups = sorted(gcounts, key=lambda g: -gcounts[g])
    vals = [gcounts[g] for g in groups]
    colors = [GROUP_COLORS.get(g, "#888888") for g in groups]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(groups)), vals, color=colors,
                  edgecolor="black", linewidth=0.5)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, fontsize=12)
    ax.set_ylabel("Unique Kinases", fontsize=13)
    ax.set_xlabel("Manning Kinome Group", fontsize=13)
    ax.set_title(
        "Kinase Group Distribution in PDB Dataset",
        fontsize=14, fontweight="bold",
    )

    for bar, c in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(c), ha="center", va="bottom",
            fontsize=10, fontweight="bold",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Histogram saved: %s", output_path)
    return fig


def _structure_group_bucket(group):
    """Collapse Other, Atypical, and Failed into a single Atypical bar."""
    if group in ("Other", "Atypical"):
        return "Atypical"
    return group


def plot_structure_histogram(resolved, n_failed=0, output_path=None):
    """Bar chart of PDB *structure* percentages per Manning group.

    Unlike ``plot_family_histogram`` (which counts unique kinases),
    this counts every PDB chain that mapped to each group.
    Other, Atypical, and failed mappings are merged into one Atypical bar.
    """
    gcounts = Counter()
    for e in resolved:
        gcounts[_structure_group_bucket(e["group"])] += 1
    if n_failed > 0:
        gcounts["Atypical"] += n_failed

    groups = sorted(gcounts, key=lambda g: -gcounts[g])
    counts = [gcounts[g] for g in groups]
    total = sum(counts)
    if total <= 0:
        raise ValueError("No structures to plot")
    pcts = [100.0 * c / total for c in counts]

    bar_color = "#5B7B91"
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(groups)),
        pcts,
        color=bar_color,
        edgecolor=bar_color,
        linewidth=0.5,
    )

    tick_font = {"fontsize": 20, "fontfamily": "Arial"}
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, **tick_font)
    ax.tick_params(axis="y", labelsize=20)
    plt.setp(ax.get_yticklabels(), fontfamily="Arial", fontsize=20)
    ax.set_ylabel("Percent of structures (%)", fontsize=26, fontfamily="Arial")
    ax.set_xlabel("Kinase family", fontsize=26, fontfamily="Arial")
    ax.set_ylim(0, max(pcts) * 1.12 + 1)
    ax.set_title(
        "PDB Structure Count per Kinome Group",
        fontsize=12, fontweight="bold",
    )

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{pct:.1f}%", ha="center", va="bottom",
            fontsize=20, fontfamily="Arial",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Structure histogram saved: %s", output_path)
    return fig


def plot_species_histogram(resolved, top_n=15, output_path=None):
    """Vertical %-bar chart of species distribution (same style as groups).

    Counts every resolved PDB+chain entry by RCSB organism name.
    Shows the top *top_n* species by count.
    """
    scounts = Counter()
    for e in resolved:
        org = (e.get("organism") or "").strip() or "Unknown"
        scounts[org] += 1

    if not scounts:
        raise ValueError("No species to plot")

    species = sorted(scounts, key=lambda s: -scounts[s])
    if top_n is not None and len(species) > top_n:
        species = species[:top_n]

    counts = [scounts[s] for s in species]
    total = sum(scounts.values())
    pcts = [100.0 * c / total for c in counts]

    bar_color = "#5B7B91"
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        range(len(species)),
        pcts,
        color=bar_color,
        edgecolor=bar_color,
        linewidth=0.5,
    )

    tick_font = {"fontsize": 14, "fontfamily": "Arial"}
    labels = [
        (s if len(s) <= 28 else s[:25] + "…") for s in species
    ]
    ax.set_xticks(range(len(species)))
    ax.set_xticklabels(labels, rotation=35, ha="right", **tick_font)
    ax.tick_params(axis="y", labelsize=20)
    plt.setp(ax.get_yticklabels(), fontfamily="Arial", fontsize=20)
    ax.set_ylabel("Percent of structures (%)", fontsize=26, fontfamily="Arial")
    ax.set_xlabel("Species", fontsize=26, fontfamily="Arial")
    ax.set_ylim(0, max(pcts) * 1.12 + 1)
    ax.set_title(
        "Distribution of Species",
        fontsize=12, fontweight="bold",
    )

    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f"{pct:.1f}%", ha="center", va="bottom",
            fontsize=14, fontfamily="Arial",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Species histogram saved: %s", output_path)
    return fig


# ═════════════════════════════════════════════════════════════════
#  6. Pipeline orchestrator
# ═════════════════════════════════════════════════════════════════

def run_pipeline(fitted_mda_dir, output_dir):
    """Execute the full PDB → CORAL mapping pipeline.

    Parameters
    ----------
    fitted_mda_dir : str | Path
        Directory containing ``{PDBID}_{Chain}[_reconstructed].pdb`` files.
    output_dir : str | Path
        Where to write ``coral_kinome.csv``, ``failed_mappings.txt``,
        and ``kinase_families.png``.

    Returns
    -------
    dict with keys: resolved, failed, coral_df, figure,
                    n_pairs, n_unique_kinases
    """
    fitted_mda_dir = Path(fitted_mda_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1 ── parse filenames
    logger.info("Step 1/6 – Parsing PDB filenames …")
    pairs = parse_pdb_filenames(fitted_mda_dir)
    logger.info("  %d PDB+chain pairs found", len(pairs))

    # 2 ── load Manning classification
    logger.info("Step 2/6 – Loading Manning kinase classification …")
    manning = load_manning_classification()
    logger.info("  %d kinase entries loaded", len(manning["by_uniprot"]))

    # 3 ── RCSB batch mapping
    logger.info("Step 3/6 – Querying RCSB GraphQL (PDB → UniProt) …")
    rcsb_results, api_fails = batch_pdb_to_uniprot(pairs)
    logger.info(
        "  RCSB done: %d mapped, %d failed",
        len(rcsb_results), len(api_fails),
    )

    # 4 ── resolve to Manning names
    logger.info("Step 4/6 – Resolving to Manning kinome names …")
    resolved, res_fails = resolve_to_manning(rcsb_results, manning)
    all_fails = api_fails + res_fails
    logger.info(
        "  %d resolved, %d total failures",
        len(resolved), len(all_fails),
    )

    # 5 ── write outputs
    logger.info("Step 5/6 – Writing CORAL CSV & failure log …")
    coral_path = output_dir / "coral_kinome.csv"
    coral_df = generate_coral_csv(resolved, coral_path)

    fail_path = output_dir / "failed_mappings.txt"
    log_failed_mappings(all_fails, fail_path)

    # 6 ── histogram
    logger.info("Step 6/6 – Generating histogram …")
    hist_path = output_dir / "kinase_families.png"
    fig = plot_structure_histogram(resolved, n_failed=len(all_fails),
                                   output_path=hist_path)

    n_unique = len({e["manning_name"] for e in resolved})
    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info("  %d PDB structures  →  %d unique kinases", len(pairs), n_unique)
    logger.info("  %d failures logged", len(all_fails))
    logger.info("  Outputs:")
    logger.info("    %s", coral_path)
    logger.info("    %s", fail_path)
    logger.info("    %s", hist_path)
    logger.info("=" * 60)

    return {
        "resolved": resolved,
        "failed": all_fails,
        "coral_df": coral_df,
        "figure": fig,
        "n_pairs": len(pairs),
        "n_unique_kinases": n_unique,
    }


def run_interpro_pipeline(
    pdb_dir="Results/InterPro_PDBs",
    output_dir="Results",
    species_top_n=15,
):
    """Annotate whole-entry InterPro PDBs and plot kinome group + species.

    Parameters
    ----------
    pdb_dir : str | Path
        Directory of ``XXXX.pdb`` files (e.g. Results/InterPro_PDBs).
    output_dir : str | Path
        Where to write CSV, failure log, and figure PNGs.
    species_top_n : int
        Number of top species bars to show.

    Returns
    -------
    dict with keys: resolved, failed, coral_df, fig_group, fig_species,
                    annot, n_pdbs, n_unique_kinases
    """
    pdb_dir = Path(pdb_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("Step 1/6 – Collecting PDB IDs from %s …", pdb_dir)
    pdb_ids = collect_pdb_ids_from_dir(pdb_dir)
    logger.info("  %d unique PDB IDs found", len(pdb_ids))

    logger.info("Step 2/6 – Loading Manning kinase classification …")
    cache_dir = Path(__file__).parent
    manning = load_manning_classification(cache_dir=cache_dir)
    logger.info("  %d kinase entries loaded", len(manning["by_uniprot"]))

    logger.info("Step 3/6 – Querying RCSB GraphQL (PDB → UniProt) …")
    rcsb_results, api_fails = batch_pdb_ids_to_uniprot(pdb_ids)
    logger.info(
        "  RCSB done: %d mapped, %d failed",
        len(rcsb_results), len(api_fails),
    )

    logger.info("Step 4/6 – Resolving to Manning kinome names …")
    resolved, res_fails = resolve_to_manning(rcsb_results, manning)
    all_fails = api_fails + res_fails
    logger.info(
        "  %d resolved, %d total failures",
        len(resolved), len(all_fails),
    )

    logger.info("Step 5/6 – Writing outputs …")
    coral_path = output_dir / "coral_kinome.csv"
    coral_df = generate_coral_csv(resolved, coral_path)
    fail_path = output_dir / "failed_mappings.txt"
    log_failed_mappings(all_fails, fail_path)

    annot = pd.DataFrame(resolved)
    annot_path = output_dir / "kinase_annotation.csv"
    annot.to_csv(annot_path, index=False)

    logger.info("Step 6/6 – Generating histograms …")
    fig_group = plot_structure_histogram(
        resolved,
        n_failed=len(all_fails),
        output_path=output_dir / "pdb_structures_per_kinome_group.png",
    )
    fig_species = plot_species_histogram(
        resolved,
        top_n=species_top_n,
        output_path=output_dir / "species_distribution.png",
    )

    n_unique = len({e["manning_name"] for e in resolved})
    logger.info("=" * 60)
    logger.info("InterPro pipeline complete!")
    logger.info(
        "  %d PDB IDs  →  %d resolved chains  →  %d unique kinases",
        len(pdb_ids), len(resolved), n_unique,
    )
    logger.info("  %d failures logged", len(all_fails))
    logger.info("=" * 60)

    return {
        "resolved": resolved,
        "failed": all_fails,
        "coral_df": coral_df,
        "fig_group": fig_group,
        "fig_species": fig_species,
        "annot": annot,
        "n_pdbs": len(pdb_ids),
        "n_unique_kinases": n_unique,
    }
