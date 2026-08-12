"""
Sequence alignment visualiser for KLIFS (85-pocket) and HMMER (hmmalign) chains.

Generates four self-contained color-coded HTML files:

    Results/MSA/klifs_passed.html     – KLIFS chains that passed DFG/APE filter
    Results/MSA/klifs_failed.html     – KLIFS chains that failed DFG/APE filter
    Results/MSA/hmmer_passed.html     – non-KLIFS chains that passed HMMER diagnostics
    Results/MSA/hmmer_failed.html     – non-KLIFS chains that failed HMMER diagnostics

Each HTML has two tab panels:
    Panel 1 – Full alignment
        KLIFS: 85-column pocket alignment (KLIFS pocket positions 1–85)
        HMMER: full kinase domain from hmmalign (~350 columns)
    Panel 2 – DFG-APE segment
        KLIFS: activation-loop sequence from activation_loop_sequences.tsv
               (left-aligned / right-padded with gaps)
        HMMER: hmmalign output trimmed to DFG…APE columns
               (DFG/APE positions found by consensus string search)

Additional per-kinase-group HTML chunks for HMMER panels:
    Results/MSA/hmmer_passed_by_group/<group>.html
    Results/MSA/hmmer_failed_by_group/<group>.html

Prerequisites
-------------
* ``klifs.build_match_residues_cache()`` written to
  ``Results/KLIFS/match_residues_cache.tsv`` before calling ``run()``.
* ``hmmalign`` on PATH (same HMMER package as ``hmmscan``).
* ``Results/HMMER/non_klifs_sequences.fasta`` exists
  (written by ``HMMERDiagnostics.run()``).
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import pandas as pd
from Bio.PDB import PDBParser, PPBuilder
from tqdm.auto import tqdm

# ── Amino-acid color scheme (Clustal-inspired) ────────────────────────────────

_AA_COLORS: Dict[str, str] = {
    # hydrophobic
    "A": "#FFFF66", "V": "#FFFF66", "I": "#FFFF66", "L": "#FFFF66",
    "M": "#FFFF66", "F": "#FFFF66", "W": "#FFFF66", "P": "#FFFF66",
    # positive
    "K": "#6699FF", "R": "#6699FF", "H": "#99BBFF",
    # negative
    "D": "#FF6666", "E": "#FF6666",
    # polar
    "S": "#33CC33", "T": "#33CC33", "N": "#66DD66", "Q": "#66DD66",
    # special
    "C": "#FF99CC", "G": "#AADDDD", "Y": "#FFAA33",
    # gap / unknown
    "-": "#EEEEEE", "X": "#CCCCCC",
}

# ── Row-header colors by failure category ─────────────────────────────────────

_ROW_COLORS: Dict[str, str] = {
    # KLIFS pass
    "klifs_pass": "#c3e6cb",
    # KLIFS DFG/APE fail reasons
    "dfg_mismatch":             "#f5c6cb",
    "no_ape_downstream":        "#ffeeba",
    "no_dfg_position":          "#fde2e4",
    "dfg_resnum_parse":         "#fde2e4",
    "dfg_resnum_not_in_chain":  "#fde2e4",
    "missing_pdb":              "#d6d8d9",
    "seq_extraction_fail":      "#d6d8d9",
    "pseudokinase":             "#bee5eb",
    # HMMER pass
    "hmmer_pass": "#b8daff",
    # HMMER flag reasons
    "flag_low_bitscore":       "#f5c6cb",
    "flag_low_model_coverage": "#ffeeba",
    "flag_short_chain":        "#fde2e4",
    "flag_high_missing_ca":    "#ffe5d0",
    "flag_span_mismatch":      "#fde2e4",
    "flag_multi_hit":          "#e8d5f5",
    "multiple_flags":          "#ffcccc",
    "no_hmmer_hit":            "#bbbbbb",
    "unknown":                 "#f8f9fa",
}

_THREE_TO_ONE: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

_EXCLUDED_FROM_FAILED_KLIFS = frozenset({
    "no_structure_id", "no_match_residues", "unexpected_klifs_columns",
})


def _seq1_safe(three: str) -> str:
    return _THREE_TO_ONE.get(three.strip().upper(), "X")


def _klifs_pos_number(pos_str: str) -> int:
    """Extract trailing integer from a KLIFS position label like 'xDFG.81' → 81."""
    m = re.search(r"(\d+)\s*$", str(pos_str))
    return int(m.group(1)) if m else 0


def _read_fasta(path: str) -> Dict[str, str]:
    """Return {header: sequence} from a FASTA file."""
    seqs: Dict[str, str] = {}
    header = ""
    parts: List[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    seqs[header] = "".join(parts)
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line)
    if header:
        seqs[header] = "".join(parts)
    return seqs


# ── HTML template helpers ─────────────────────────────────────────────────────

_CSS = """\
body{font-family:monospace;font-size:11px;margin:8px}
.tabs{display:flex;gap:4px;margin-bottom:6px}
.tab-btn{padding:5px 14px;cursor:pointer;border:1px solid #aaa;
         background:#eee;border-radius:3px 3px 0 0}
.tab-btn.active{background:#fff;border-bottom:1px solid #fff;font-weight:bold}
.panel{display:none;overflow:auto;max-height:85vh;border:1px solid #ccc;padding:4px}
.panel.active{display:block}
table{border-collapse:collapse;white-space:nowrap}
th{position:sticky;top:0;background:#ddd;padding:1px 3px;
   border:1px solid #bbb;z-index:2;font-size:10px}
th.rowlabel{left:0;z-index:3}
td{padding:0 1px;text-align:center;min-width:13px;border:1px solid #ddd}
td.rowlabel{text-align:left;padding:0 4px;white-space:nowrap;
            position:sticky;left:0;z-index:1}
.legend{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0}
.legend-item{padding:2px 8px;border:1px solid #aaa;border-radius:3px;font-size:11px}
"""

_JS = """\
function showTab(fileId, tabId){
  document.querySelectorAll('#'+fileId+' .panel').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('#'+fileId+' .tab-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById(tabId).classList.add('active');
  event.target.classList.add('active');
}
"""


def _aa_cell(aa: str) -> str:
    c = _AA_COLORS.get(aa.upper(), _AA_COLORS["X"])
    return f'<td style="background:{c}">{aa}</td>'


def _render_aln_table(
    names: List[str],
    seqs: Dict[str, str],
    col_labels: Optional[List[str]],
    row_meta: Dict[str, Tuple[str, str]],  # name -> (category, group)
) -> str:
    """Render an HTML <table> for the alignment."""
    n_cols = max((len(s) for s in seqs.values()), default=0)
    labels = col_labels if col_labels else [str(i + 1) for i in range(n_cols)]

    rows_html: List[str] = ["<table>"]
    # Header row
    header_cells = "".join(
        f'<th title="{lbl}">{lbl if len(str(lbl)) <= 5 else str(lbl)[-4:]}</th>'
        for lbl in labels
    )
    rows_html.append(
        f"<tr><th class='rowlabel'>ID</th>"
        f"<th>Category</th>{header_cells}</tr>"
    )
    for name in names:
        seq = seqs.get(name, "")
        seq = seq.ljust(n_cols, "-")
        cat, grp = row_meta.get(name, ("unknown", ""))
        row_color = _ROW_COLORS.get(cat, _ROW_COLORS["unknown"])
        cells = "".join(_aa_cell(aa) for aa in seq)
        rows_html.append(
            f"<tr>"
            f"<td class='rowlabel' style='background:{row_color}' title='{grp}'>{name}</td>"
            f"<td style='background:{row_color};white-space:nowrap'>{cat}</td>"
            f"{cells}</tr>"
        )
    rows_html.append("</table>")
    return "\n".join(rows_html)


def _legend_html(categories: List[str]) -> str:
    items = []
    for cat in sorted(set(categories)):
        c = _ROW_COLORS.get(cat, _ROW_COLORS["unknown"])
        items.append(f"<span class='legend-item' style='background:{c}'>{cat}</span>")
    return "<div class='legend'>" + "".join(items) + "</div>"


def _write_html_file(
    path: str,
    title: str,
    names: List[str],
    full_seqs: Dict[str, str],
    full_col_labels: Optional[List[str]],
    dfg_ape_seqs: Dict[str, str],
    dfg_ape_col_labels: Optional[List[str]],
    row_meta: Dict[str, Tuple[str, str]],  # name -> (category, group)
) -> None:
    """Write a self-contained two-tab HTML MSA file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    cats = [row_meta.get(n, ("unknown", ""))[0] for n in names]
    legend = _legend_html(cats)

    file_id = re.sub(r"[^a-zA-Z0-9]", "_", os.path.basename(path))
    tab1_id = f"{file_id}_t1"
    tab2_id = f"{file_id}_t2"

    full_table = _render_aln_table(names, full_seqs, full_col_labels, row_meta)
    dfg_table  = _render_aln_table(names, dfg_ape_seqs, dfg_ape_col_labels, row_meta)

    html = f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{_CSS}</style>
<script>{_JS}</script>
</head><body>
<h2>{title} — {len(names)} sequences</h2>
{legend}
<div id="{file_id}">
  <div class="tabs">
    <button class="tab-btn active"
      onclick="showTab('{file_id}','{tab1_id}')">Full alignment</button>
    <button class="tab-btn"
      onclick="showTab('{file_id}','{tab2_id}')">DFG-APE segment</button>
  </div>
  <div id="{tab1_id}" class="panel active">{full_table}</div>
  <div id="{tab2_id}" class="panel">{dfg_table}</div>
</div>
</body></html>"""

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    kb = os.path.getsize(path) // 1024
    print(f"  Wrote {path}  ({len(names)} rows, {kb} KB)")


# ── Main class ────────────────────────────────────────────────────────────────

class MSAVisualiser:
    """Generate color-coded HTML MSA files from KLIFS and HMMER results.

    Parameters
    ----------
    klifs_dir, hmmer_dir : str
        Directories containing KLIFS and HMMER outputs.
    chain_dir : str
        Directory with all extracted chain PDB files (InterPro_protein_chains).
    motif_dir : str
        Directory with motif-filtered chain PDB files (for KLIFS passed chains).
    output_dir : str
        Root directory for generated HTML files.
    annot_csv : str
        kinase_annotation_all_chains.csv — used for kinase-group lookup.
    loop_tsv : str
        activation_loop_sequences.tsv — loop sequences for KLIFS-passed panel.
    hmm_path : str
        Path to Pkinase.hmm (used by hmmalign).
    hmmer_fasta : str
        non_klifs_sequences.fasta (full chain sequences for HMMER chains).
    diagnostics_tsv : str
        hmmer_diagnostics.tsv — flag info for HMMER chains.
    failures_txt : str
        hmmer_failures.txt — flagged HMMER basenames.
    klifs_failures_txt : str
        klifs_filter_failures.txt — KLIFS failure reasons.
    hmmalign_binary : str
        Name/path of the hmmalign executable.
    """

    def __init__(
        self,
        klifs_dir: str = "Results/KLIFS",
        hmmer_dir: str = "Results/HMMER",
        chain_dir: str = "Results/InterPro_protein_chains/",
        motif_dir: str = "Results/motif_filtered_chains/",
        output_dir: str = "Results/MSA",
        annot_csv: str = "Results/kinase_annotation_all_chains.csv",
        loop_tsv: str = "Results/activation_loop_sequences.tsv",
        hmm_path: str = "Results/HMMER/Pkinase.hmm",
        hmmer_fasta: str = "Results/HMMER/non_klifs_sequences.fasta",
        diagnostics_tsv: str = "Results/HMMER/hmmer_diagnostics.tsv",
        failures_txt: str = "Results/HMMER/hmmer_failures.txt",
        klifs_failures_txt: str = "Results/KLIFS/klifs_filter_failures.txt",
        hmmalign_binary: str = "hmmalign",
    ):
        self.klifs_dir = klifs_dir
        self.hmmer_dir = hmmer_dir
        self.chain_dir = chain_dir
        self.motif_dir = motif_dir
        self.output_dir = output_dir
        self.annot_csv = annot_csv
        self.loop_tsv = loop_tsv
        self.hmm_path = hmm_path
        self.hmmer_fasta = hmmer_fasta
        self.diagnostics_tsv = diagnostics_tsv
        self.failures_txt = failures_txt
        self.klifs_failures_txt = klifs_failures_txt
        self.hmmalign_binary = hmmalign_binary

    # ── PDB residue lookup ────────────────────────────────────────────────

    def _load_residue_dict(self, pdb_path: str) -> Dict[int, str]:
        """Return {pdb_resnum: one_letter_code} for ATOM residues in *pdb_path*."""
        parser = PDBParser(QUIET=True)
        ppb = PPBuilder()
        try:
            structure = parser.get_structure("x", pdb_path)
        except Exception:
            return {}
        result: Dict[int, str] = {}
        for pp in ppb.build_peptides(structure):
            for residue in pp:
                resnum = residue.id[1]
                result[resnum] = _seq1_safe(residue.resname)
        return result

    def _find_pdb(self, basename: str) -> Optional[str]:
        """Find the PDB file for *basename*, checking motif_dir then chain_dir."""
        for d in (self.motif_dir, self.chain_dir):
            p = os.path.join(d, basename + ".pdb")
            if os.path.isfile(p):
                return p
        return None

    # ── Metadata helpers ──────────────────────────────────────────────────

    def _load_annotation(self) -> Dict[str, str]:
        """Return {basename_without_ext: manning_group or group or ''} from annot CSV."""
        if not os.path.isfile(self.annot_csv):
            return {}
        df = pd.read_csv(self.annot_csv)
        result: Dict[str, str] = {}
        for _, row in df.iterrows():
            pdb_file = str(row.get("pdb_file", ""))
            base = os.path.splitext(os.path.basename(pdb_file))[0]
            group = str(row.get("manning_group", "") or row.get("group", "") or "")
            result[base] = group if group not in ("nan", "None", "") else "unknown"
        return result

    def _get_klifs_passed_basenames(self) -> List[str]:
        """Basenames from activation_loop_sequences.tsv where klifs_structure_id is set."""
        if not os.path.isfile(self.loop_tsv):
            return []
        df = pd.read_csv(self.loop_tsv, sep="\t")
        if "klifs_structure_id" not in df.columns:
            return []
        mask = df["klifs_structure_id"].notna() & (df["klifs_structure_id"] != "")
        return df.loc[mask, "pdb_basename"].tolist()

    def _get_klifs_failed_basenames(self) -> Dict[str, str]:
        """Return {basename: reason} for includable KLIFS-failed chains."""
        if not os.path.isfile(self.klifs_failures_txt):
            return {}
        result: Dict[str, str] = {}
        with open(self.klifs_failures_txt, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                basename, reason = parts[0].strip(), parts[1].strip()
                if reason not in _EXCLUDED_FROM_FAILED_KLIFS:
                    result[basename] = reason
        return result

    def _get_hmmer_passed_basenames(self) -> List[str]:
        """Non-flagged HMMER chains: in diagnostics TSV with any_flag == False."""
        if not os.path.isfile(self.diagnostics_tsv):
            return []
        df = pd.read_csv(self.diagnostics_tsv, sep="\t")
        if "any_flag" not in df.columns:
            return df["basename"].tolist()
        return df.loc[~df["any_flag"].fillna(True).astype(bool), "basename"].tolist()

    def _get_hmmer_failed_basenames(self) -> Dict[str, str]:
        """Return {basename: primary_flag_reason} for HMMER-failed chains (had a hit)."""
        if not os.path.isfile(self.diagnostics_tsv):
            return {}
        df = pd.read_csv(self.diagnostics_tsv, sep="\t")
        failed = df[df["any_flag"].fillna(True).astype(bool)].copy()
        flag_cols = [c for c in df.columns if c.startswith("flag_")]
        result: Dict[str, str] = {}
        for _, row in failed.iterrows():
            active = [c for c in flag_cols if row.get(c, False)]
            reason = active[0] if len(active) == 1 else ("multiple_flags" if active else "unknown")
            result[str(row["basename"])] = reason
        return result

    # ── KLIFS 85-column alignment ─────────────────────────────────────────

    def _build_klifs_matrix(
        self, basenames: List[str]
    ) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
        """Build 85-pocket alignment for *basenames* from the match_residues cache.

        Returns
        -------
        matrix : {basename: {klifs_pos_label: one_letter_aa_or_gap}}
        ordered_positions : list of KLIFS position labels sorted by position number
        """
        cache_path = os.path.join(self.klifs_dir, "match_residues_cache.tsv")
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(
                f"Match-residues cache not found: {cache_path}. "
                "Run klifs.build_match_residues_cache() first."
            )
        cache = pd.read_csv(cache_path, sep="\t")
        cache_bases = set(cache["basename"].astype(str))

        all_positions = sorted(
            cache["klifs_position"].dropna().unique(),
            key=_klifs_pos_number,
        )

        matrix: Dict[str, Dict[str, str]] = {}
        for basename in tqdm(basenames, desc="KLIFS matrix"):
            if basename not in cache_bases:
                matrix[basename] = {p: "-" for p in all_positions}
                continue
            rows = cache[cache["basename"] == basename]
            pdb_path = self._find_pdb(basename)
            res_dict = self._load_residue_dict(pdb_path) if pdb_path else {}

            row_dict: Dict[str, str] = {}
            pos_to_xray: Dict[str, Optional[int]] = {
                str(r["klifs_position"]): (int(r["xray_position"])
                                           if pd.notna(r["xray_position"]) else None)
                for _, r in rows.iterrows()
            }
            for pos in all_positions:
                xray = pos_to_xray.get(pos)
                if xray is None:
                    row_dict[pos] = "-"
                else:
                    row_dict[pos] = res_dict.get(xray, "X")
            matrix[basename] = row_dict

        return matrix, all_positions

    @staticmethod
    def _matrix_to_seqs(
        matrix: Dict[str, Dict[str, str]],
        positions: List[str],
    ) -> Dict[str, str]:
        """Flatten the KLIFS matrix to {basename: sequence_string}."""
        return {
            name: "".join(row.get(p, "-") for p in positions)
            for name, row in matrix.items()
        }

    # ── KLIFS DFG-APE panel (loop sequences from TSV) ────────────────────

    def _build_klifs_loop_seqs(
        self, passed_names: List[str]
    ) -> Tuple[Dict[str, str], int]:
        """Return padded loop sequences for KLIFS-passed chains.

        Sequences are left-aligned at DFG+1 and right-padded to max length.
        Returns (padded_seqs, max_len).
        """
        if not os.path.isfile(self.loop_tsv):
            return {n: "" for n in passed_names}, 0
        loop_df = pd.read_csv(self.loop_tsv, sep="\t")
        loop_map = dict(zip(loop_df["pdb_basename"], loop_df["loop_sequence"].fillna("")))
        seqs = {n: str(loop_map.get(n, "")) for n in passed_names}
        max_len = max((len(s) for s in seqs.values()), default=0)
        padded = {n: s.ljust(max_len, "-") for n, s in seqs.items()}
        return padded, max_len

    def _build_klifs_failed_dfg_ape(
        self,
        failed_names: List[str],
        matrix: Dict[str, Dict[str, str]],
        positions: List[str],
    ) -> Dict[str, str]:
        """For KLIFS-failed chains show positions 81 onward (DFG and downstream)."""
        dfg_start = next(
            (i for i, p in enumerate(positions) if _klifs_pos_number(p) >= 81),
            0,
        )
        dfg_positions = positions[dfg_start:]
        return {
            name: "".join(matrix.get(name, {}).get(p, "-") for p in dfg_positions)
            for name in failed_names
        }

    # ── HMMER alignment (hmmalign) ────────────────────────────────────────

    def _build_hmmer_fasta(self, basenames: List[str], label: str) -> str:
        """Write a FASTA for *basenames* from the non-KLIFS FASTA; return temp path."""
        all_seqs = _read_fasta(self.hmmer_fasta) if os.path.isfile(self.hmmer_fasta) else {}
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".fasta", prefix=f"msa_{label}_",
            delete=False, encoding="utf-8"
        )
        n_written = 0
        for basename in basenames:
            seq = all_seqs.get(basename)
            if seq:
                tmp.write(f">{basename}\n{seq}\n")
                n_written += 1
        tmp.close()
        print(f"  HMMER FASTA ({label}): {n_written}/{len(basenames)} sequences written")
        return tmp.name

    def _run_hmmalign(self, fasta_path: str) -> Dict[str, str]:
        """Run hmmalign and return {seq_id: gapped_sequence}."""
        with tempfile.NamedTemporaryFile(suffix=".afa", delete=False) as tmp:
            out_path = tmp.name
        try:
            result = subprocess.run(
                [
                    self.hmmalign_binary,
                    "--trim",
                    "--outformat", "afa",
                    self.hmm_path,
                    fasta_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"hmmalign not found. Install HMMER: conda install -c bioconda hmmer"
            ) from exc

        if result.returncode != 0:
            raise RuntimeError(
                f"hmmalign failed (exit {result.returncode}):\n{result.stderr[:500]}"
            )

        aligned: Dict[str, str] = {}
        header = ""
        parts: List[str] = []
        for line in result.stdout.splitlines():
            if line.startswith(">"):
                if header:
                    aligned[header] = "".join(parts).upper()
                header = line[1:].split()[0]
                parts = []
            else:
                parts.append(line.strip())
        if header:
            aligned[header] = "".join(parts).upper()

        if os.path.isfile(out_path):
            os.remove(out_path)
        return aligned

    # ── DFG/APE column detection ──────────────────────────────────────────

    @staticmethod
    def _consensus(aligned: Dict[str, str]) -> str:
        """Most-frequent non-gap character at each column."""
        seqs = list(aligned.values())
        if not seqs:
            return ""
        n_cols = max(len(s) for s in seqs)
        consensus: List[str] = []
        for col in range(n_cols):
            counts: Dict[str, int] = defaultdict(int)
            for seq in seqs:
                c = seq[col].upper() if col < len(seq) else "-"
                if c not in ("-", "."):
                    counts[c] += 1
            consensus.append(max(counts, key=counts.get) if counts else "-")
        return "".join(consensus)

    def find_dfg_ape_columns(
        self, aligned: Dict[str, str]
    ) -> Tuple[int, int]:
        """Find the start column of DFG and the end column of APE in the alignment.

        Searches the consensus sequence for the conserved 'DFG' triplet (in the
        C-terminal half) then the first 'APE' triplet downstream.

        Returns (dfg_col, ape_end_col) as 0-based column indices into the alignment.
        Raises ``ValueError`` if either motif cannot be found.
        """
        cons = self._consensus(aligned)
        mid = len(cons) // 2

        dfg_col = cons.find("DFG", mid)
        if dfg_col == -1:
            dfg_col = cons.find("DFG")
        if dfg_col == -1:
            raise ValueError("DFG motif not found in alignment consensus.")

        ape_col = cons.find("APE", dfg_col + 3)
        if ape_col == -1:
            # Fallback: use a reasonable window after DFG
            ape_col = min(dfg_col + 45, len(cons) - 3)
            print(
                f"  WARNING: APE not found in consensus after DFG col {dfg_col}; "
                f"using col {ape_col} as APE estimate."
            )
        ape_end_col = ape_col + 3  # include the 'E' of APE
        return dfg_col, ape_end_col

    @staticmethod
    def _trim_to_cols(
        aligned: Dict[str, str],
        start_col: int,
        end_col: int,
    ) -> Dict[str, str]:
        return {name: seq[start_col:end_col] for name, seq in aligned.items()}

    # ── HTML orchestration ────────────────────────────────────────────────

    def _write_klifs_html(
        self,
        category: str,
        names: List[str],
        full_seqs: Dict[str, str],
        col_labels: List[str],
        dfg_ape_seqs: Dict[str, str],
        dfg_ape_labels: Optional[List[str]],
        row_meta: Dict[str, Tuple[str, str]],
        output_path: str,
    ) -> None:
        title = f"KLIFS {'passed' if category == 'passed' else 'failed'} — 85-pocket alignment"
        _write_html_file(
            output_path, title, names,
            full_seqs, col_labels,
            dfg_ape_seqs, dfg_ape_labels,
            row_meta,
        )

    def _write_hmmer_html(
        self,
        category: str,
        names: List[str],
        full_aligned: Dict[str, str],
        dfg_col: int,
        ape_end_col: int,
        col_labels: List[str],
        row_meta: Dict[str, Tuple[str, str]],
        output_path: str,
    ) -> None:
        dfg_ape_seqs = self._trim_to_cols(full_aligned, dfg_col, ape_end_col)
        dfg_ape_labels = col_labels[dfg_col:ape_end_col]
        title = f"HMMER {'passed' if category == 'passed' else 'failed'} — hmmalign kinase domain"
        _write_html_file(
            output_path, title, names,
            full_aligned, col_labels,
            dfg_ape_seqs, dfg_ape_labels,
            row_meta,
        )

    def _write_hmmer_group_chunks(
        self,
        category: str,
        names: List[str],
        full_aligned: Dict[str, str],
        dfg_col: int,
        ape_end_col: int,
        col_labels: List[str],
        row_meta: Dict[str, Tuple[str, str]],
        chunk_dir: str,
    ) -> None:
        groups: Dict[str, List[str]] = defaultdict(list)
        for name in names:
            grp = row_meta.get(name, ("", "unknown"))[1] or "unknown"
            groups[grp].append(name)
        os.makedirs(chunk_dir, exist_ok=True)
        for grp, grp_names in sorted(groups.items()):
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", grp)
            path = os.path.join(chunk_dir, f"{safe}.html")
            self._write_hmmer_html(
                category, grp_names, full_aligned,
                dfg_col, ape_end_col, col_labels, row_meta, path,
            )

    # ── Public run() ─────────────────────────────────────────────────────

    def run(self) -> Dict[str, str]:
        """Build all MSA HTML files. Returns a dict of {label: path}."""
        os.makedirs(self.output_dir, exist_ok=True)
        annotation = self._load_annotation()
        output_paths: Dict[str, str] = {}

        # ── KLIFS passed ──────────────────────────────────────────────────
        print("── KLIFS passed ──")
        kp_names = self._get_klifs_passed_basenames()
        print(f"  {len(kp_names)} passed KLIFS chains")
        kp_matrix, kp_positions = self._build_klifs_matrix(kp_names)
        kp_full_seqs = self._matrix_to_seqs(kp_matrix, kp_positions)
        kp_loop_seqs, _ = self._build_klifs_loop_seqs(kp_names)
        kp_meta = {
            name: ("klifs_pass", annotation.get(name, "unknown"))
            for name in kp_names
        }
        kp_path = os.path.join(self.output_dir, "klifs_passed.html")
        self._write_klifs_html(
            "passed", kp_names,
            kp_full_seqs, [str(p) for p in kp_positions],
            kp_loop_seqs, None,
            kp_meta, kp_path,
        )
        output_paths["klifs_passed"] = kp_path

        # ── KLIFS failed ──────────────────────────────────────────────────
        print("── KLIFS failed ──")
        kf_dict = self._get_klifs_failed_basenames()
        kf_names = list(kf_dict.keys())
        print(f"  {len(kf_names)} failed KLIFS chains (with match_residues data)")
        kf_matrix, kf_positions = self._build_klifs_matrix(kf_names)
        kf_full_seqs = self._matrix_to_seqs(kf_matrix, kf_positions)
        kf_dfg_seqs = self._build_klifs_failed_dfg_ape(kf_names, kf_matrix, kf_positions)
        kf_meta = {
            name: (kf_dict.get(name, "unknown"), annotation.get(name, "unknown"))
            for name in kf_names
        }
        # DFG-APE column labels: positions 81 onwards
        dfg_start_idx = next(
            (i for i, p in enumerate(kf_positions) if _klifs_pos_number(p) >= 81), 0
        )
        kf_dfg_labels = [str(p) for p in kf_positions[dfg_start_idx:]]
        kf_path = os.path.join(self.output_dir, "klifs_failed.html")
        self._write_klifs_html(
            "failed", kf_names,
            kf_full_seqs, [str(p) for p in kf_positions],
            kf_dfg_seqs, kf_dfg_labels,
            kf_meta, kf_path,
        )
        output_paths["klifs_failed"] = kf_path

        # ── HMMER passed ──────────────────────────────────────────────────
        print("── HMMER passed ──")
        hp_names = self._get_hmmer_passed_basenames()
        print(f"  {len(hp_names)} passed HMMER chains")
        hp_fasta = self._build_hmmer_fasta(hp_names, "hmmer_passed")
        hp_aligned = self._run_hmmalign(hp_fasta)
        hp_names_aln = [n for n in hp_names if n in hp_aligned]
        hp_dfg_col, hp_ape_end = self.find_dfg_ape_columns(hp_aligned)
        n_cols_hp = max((len(s) for s in hp_aligned.values()), default=0)
        hp_col_labels = [str(i + 1) for i in range(n_cols_hp)]
        hp_meta = {
            name: ("hmmer_pass", annotation.get(name, "unknown"))
            for name in hp_names_aln
        }
        hp_path = os.path.join(self.output_dir, "hmmer_passed.html")
        self._write_hmmer_html(
            "passed", hp_names_aln, hp_aligned,
            hp_dfg_col, hp_ape_end, hp_col_labels, hp_meta, hp_path,
        )
        output_paths["hmmer_passed"] = hp_path
        hp_chunk_dir = os.path.join(self.output_dir, "hmmer_passed_by_group")
        self._write_hmmer_group_chunks(
            "passed", hp_names_aln, hp_aligned,
            hp_dfg_col, hp_ape_end, hp_col_labels, hp_meta, hp_chunk_dir,
        )
        output_paths["hmmer_passed_by_group"] = hp_chunk_dir

        # ── HMMER failed ──────────────────────────────────────────────────
        print("── HMMER failed ──")
        hf_dict = self._get_hmmer_failed_basenames()
        hf_names = list(hf_dict.keys())
        print(f"  {len(hf_names)} failed HMMER chains (had a hit but flagged)")
        hf_fasta = self._build_hmmer_fasta(hf_names, "hmmer_failed")
        hf_aligned = self._run_hmmalign(hf_fasta)
        hf_names_aln = [n for n in hf_names if n in hf_aligned]
        hf_dfg_col, hf_ape_end = self.find_dfg_ape_columns(hf_aligned)
        n_cols_hf = max((len(s) for s in hf_aligned.values()), default=0)
        hf_col_labels = [str(i + 1) for i in range(n_cols_hf)]
        hf_meta = {
            name: (hf_dict.get(name, "unknown"), annotation.get(name, "unknown"))
            for name in hf_names_aln
        }
        hf_path = os.path.join(self.output_dir, "hmmer_failed.html")
        self._write_hmmer_html(
            "failed", hf_names_aln, hf_aligned,
            hf_dfg_col, hf_ape_end, hf_col_labels, hf_meta, hf_path,
        )
        output_paths["hmmer_failed"] = hf_path
        hf_chunk_dir = os.path.join(self.output_dir, "hmmer_failed_by_group")
        self._write_hmmer_group_chunks(
            "failed", hf_names_aln, hf_aligned,
            hf_dfg_col, hf_ape_end, hf_col_labels, hf_meta, hf_chunk_dir,
        )
        output_paths["hmmer_failed_by_group"] = hf_chunk_dir

        # Cleanup temp FASTA files
        for tmp in (hp_fasta, hf_fasta):
            try:
                os.remove(tmp)
            except OSError:
                pass

        print(f"\nAll MSA HTMLs written to {self.output_dir}/")
        return output_paths
