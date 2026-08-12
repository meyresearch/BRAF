"""
KinHub (Manning) kinase classification — same source and resolution order as
``ManningTree/plotFamilies.py``: UniProt ID match, then entry-name mnemonic,
then batch UniProt gene lookup for non-human accessions.

Used to add ``manning_name``, ``manning_group``, ``manning_family`` columns to
per-chain annotation tables.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

logger = logging.getLogger(__name__)

KINHUB_CSV_URL = (
    "https://raw.githubusercontent.com/openkinome/kinodata/"
    "master/data/KinHubKinaseList.csv"
)


def load_manning_classification(cache_dir: str | Path | None = None) -> dict[str, Any]:
    """Load KinHub CSV and return lookup dicts (by_uniprot, by_hgnc, by_manning, by_xname)."""
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parent
    cache_file = Path(cache_dir) / "kinhub_kinases.csv"

    csv_text: str | None = None
    if cache_file.exists():
        csv_text = cache_file.read_text()
        logger.info("Manning KinHub classification loaded from cache: %s", cache_file)
    else:
        try:
            resp = requests.get(KINHUB_CSV_URL, timeout=30)
            resp.raise_for_status()
            csv_text = resp.text
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(csv_text)
            logger.info("Downloaded KinHub CSV to %s", cache_file)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot fetch KinHub data ({exc}). Download manually from "
                f"{KINHUB_CSV_URL} and save as {cache_file}"
            ) from exc

    return _parse_kinhub_csv(csv_text)


def _parse_kinhub_csv(csv_text: str) -> dict[str, Any]:
    by_uniprot: dict = {}
    by_hgnc: dict = {}
    by_manning: dict = {}
    by_xname: dict = {}

    csv_text = csv_text.replace("\u00a0", " ")
    for row in csv.DictReader(io.StringIO(csv_text)):
        entry = {
            "manning_name": row["Manning Name"].strip(),
            "hgnc_name": (row.get("HGNC Name") or "").strip(),
            "xname": (row.get("xName") or "").strip(),
            "group": row["Group"].strip(),
            "family": (row.get("Family") or "").strip(),
            "subfamily": (row.get("SubFamily") or "").strip(),
            "uniprot_id": (row.get("UniprotID") or "").strip(),
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
        "by_hgnc": by_hgnc,
        "by_manning": by_manning,
        "by_xname": by_xname,
    }


def _manning_match_gene_token(manning: dict[str, Any], token: str) -> dict | None:
    if not token:
        return None
    gu = token.strip().upper()
    if not gu:
        return None
    return (
        manning["by_hgnc"].get(gu)
        or manning["by_manning"].get(gu)
        or manning["by_xname"].get(gu)
    )


def _uniprot_taxon_id(js: dict | None) -> int | None:
    if not js:
        return None
    org = js.get("organism") or {}
    tid = org.get("taxonId")
    return int(tid) if tid is not None else None


def _uniprot_entry_name_strings(js: dict | None) -> list[str]:
    """Strings like ``BRAF_HUMAN`` used as RCSB-style entry names for mnemonic match."""
    if not js:
        return []
    out: list[str] = []
    ukb = js.get("uniProtkbId")
    if isinstance(ukb, str) and ukb.strip():
        out.append(ukb.strip())
    return out


def _uniprot_gene_strings(js: dict | None) -> list[str]:
    if not js:
        return []
    names: list[str] = []
    for g in js.get("genes") or []:
        if not isinstance(g, dict):
            continue
        gn = g.get("geneName") or {}
        v = gn.get("value") if isinstance(gn, dict) else None
        if v:
            names.append(str(v))
        for syn in g.get("synonyms") or []:
            if not isinstance(syn, dict):
                continue
            sv = syn.get("value")
            if sv:
                names.append(str(sv))
    return names


def _resolve_hit_from_entry_names(
    manning: dict[str, Any], entry_names: list[str]
) -> dict | None:
    for en in entry_names:
        parts = str(en).split("_")
        if len(parts) >= 2:
            mnem = parts[0].upper()
            hit = _manning_match_gene_token(manning, mnem)
            if hit:
                return hit
    return None


def batch_uniprot_genes(
    uids: list[str],
    *,
    session: requests.Session | None = None,
    batch_size: int = 200,
) -> dict[str, list[str]]:
    """Fetch gene names from UniProt REST (same query shape as plotFamilies)."""
    sess = session or requests
    result: dict[str, list[str]] = {}
    total = len(uids)
    for i in range(0, total, batch_size):
        batch = uids[i : i + batch_size]
        query = " OR ".join(f"accession:{u}" for u in batch)
        try:
            resp = sess.get(
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
                names: list[str] = []
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
            logger.error("UniProt batch gene error: %s", exc)

        if i + batch_size < total:
            time.sleep(0.5)
    return result


def enrich_annotation_dataframe(
    annot: pd.DataFrame,
    normalize_uniprot: Callable[[str], str],
    fetch_uniprot_json: Callable[[str], dict | None],
    session: requests.Session,
    *,
    kinhub_cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Add ``manning_name``, ``manning_group``, ``manning_family`` using KinHub + Manning
    resolution (aligned with ``plotFamilies.resolve_to_manning``).

    Expects columns ``uniprot_acc`` and optionally ``gene``.
    """
    out = annot.copy()
    for col in ("manning_name", "manning_group", "manning_family"):
        if col not in out.columns:
            out[col] = pd.NA

    if "uniprot_acc" not in out.columns:
        logger.warning("enrich_annotation_dataframe: no uniprot_acc column; skipping KinHub")
        return out

    manning = load_manning_classification(kinhub_cache_dir)

    roots_series = out["uniprot_acc"].map(
        lambda x: normalize_uniprot(x) if isinstance(x, str) and x.strip() else pd.NA
    )
    unique_roots = sorted({r for r in roots_series.dropna().unique() if isinstance(r, str)})

    json_by_root: dict[str, dict | None] = {}
    for r in unique_roots:
        json_by_root[r] = fetch_uniprot_json(r)

    # Representative gene string per root (first non-empty in table)
    gene_by_root: dict[str, str | None] = {}
    if "gene" in out.columns:
        for r in unique_roots:
            mask = roots_series == r
            genes = (
                out.loc[mask, "gene"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            genes = genes[genes.str.len() > 0]
            gene_by_root[r] = genes.iloc[0] if len(genes) else None
    else:
        gene_by_root = {r: None for r in unique_roots}

    hit_by_root: dict[str, dict | None] = {r: None for r in unique_roots}
    nonhuman_roots_set: set[str] = set()

    for r in unique_roots:
        js = json_by_root.get(r)
        hit = manning["by_uniprot"].get(r)
        if hit:
            hit_by_root[r] = hit
            continue

        enames = _uniprot_entry_name_strings(js)
        hit = _resolve_hit_from_entry_names(manning, enames)
        if hit:
            hit_by_root[r] = hit
            continue

        g_row = gene_by_root.get(r)
        hit = _manning_match_gene_token(manning, g_row) if g_row else None
        if hit:
            hit_by_root[r] = hit
            continue

        for gn in _uniprot_gene_strings(js):
            hit = _manning_match_gene_token(manning, gn)
            if hit:
                hit_by_root[r] = hit
                break
        if hit_by_root[r]:
            continue

        tax = _uniprot_taxon_id(js)
        if tax is not None and tax != 9606:
            nonhuman_roots_set.add(r)
        else:
            hit_by_root[r] = None

    nonhuman_roots = sorted(nonhuman_roots_set)
    if nonhuman_roots:
        logger.info(
            "KinHub: resolving %d non-human UniProt accessions via gene batch lookup",
            len(nonhuman_roots),
        )
        gene_lookup = batch_uniprot_genes(nonhuman_roots, session=session)
        for r in nonhuman_roots:
            if hit_by_root.get(r):
                continue
            hit = None
            for g in gene_lookup.get(r, []):
                hit = _manning_match_gene_token(manning, g)
                if hit:
                    break
            hit_by_root[r] = hit

    def _root_to_hit(r: Any) -> dict | None:
        if not isinstance(r, str):
            return None
        return hit_by_root.get(r)

    mapped_hits = roots_series.map(_root_to_hit)
    out["manning_name"] = mapped_hits.map(lambda h: h["manning_name"] if h else pd.NA)
    out["manning_group"] = mapped_hits.map(lambda h: h["group"] if h else pd.NA)
    out["manning_family"] = mapped_hits.map(
        lambda h: h["family"] if h and h.get("family") else pd.NA
    )

    return out
