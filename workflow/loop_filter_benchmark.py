"""Helpers for the August 03c three-way activation-loop filter benchmark."""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from workflow.msa_visualiser import _write_html_file
from workflow.utilities import extract_sequence_from_pdb


def extract_dfg_ape_loops_from_pdbs(
    chain_dir: str,
    basenames: Optional[Sequence[str]] = None,
    loop_tsv: str = "activation_loop_sequences.tsv",
    *,
    dfg_motif: str = "DFG",
    ape_motif: str = "APE",
) -> pd.DataFrame:
    """
    Extract DFG→APE loop sequences (residues *between* the motifs) from PDBs.

    Matches KLIFS/HMMER loop definition: ``seq[dfg_end:ape_start]``.
    Writes a TSV with columns ``pdb_basename``, ``loop_sequence``.
    """
    chain_dir = chain_dir.rstrip("/") + "/"
    if basenames is None:
        basenames = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(chain_dir)
            if f.endswith(".pdb")
        )

    records = []
    n_fail = 0
    for name in tqdm(list(basenames), desc="Extract DFG–APE loops"):
        stem = os.path.splitext(os.path.basename(str(name)))[0]
        pdb_path = os.path.join(chain_dir, f"{stem}.pdb")
        if not os.path.isfile(pdb_path):
            n_fail += 1
            continue
        seq = extract_sequence_from_pdb(pdb_path)
        if not seq:
            n_fail += 1
            continue
        dfg_idx = seq.find(dfg_motif)
        if dfg_idx < 0:
            n_fail += 1
            continue
        ape_idx = seq.find(ape_motif, dfg_idx + len(dfg_motif))
        if ape_idx < 0:
            n_fail += 1
            continue
        loop_seq = seq[dfg_idx + len(dfg_motif) : ape_idx]
        records.append({"pdb_basename": stem, "loop_sequence": loop_seq})

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(os.path.abspath(loop_tsv)) or ".", exist_ok=True)
    df.to_csv(loop_tsv, sep="\t", index=False)
    print(
        f"Wrote {len(df)} loop sequences → {loop_tsv} "
        f"(failed/missing: {n_fail})"
    )
    return df


def load_loop_lengths(loop_tsv: str) -> np.ndarray:
    """Return loop lengths from an activation-loop TSV."""
    df = pd.read_csv(loop_tsv, sep="\t")
    if "loop_sequence" not in df.columns:
        raise KeyError(f"{loop_tsv} missing loop_sequence column")
    return df["loop_sequence"].astype(str).str.len().to_numpy()


def plot_loop_length_histograms(
    method_to_loop_tsv: Dict[str, str],
    save_path: str,
    *,
    bins: int = 50,
    show: bool = True,
) -> str:
    """
    Plot overlaid + per-method histograms of DFG–APE loop lengths.

    ``method_to_loop_tsv`` maps display name → path to loop TSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)

    series = {}
    for label, path in method_to_loop_tsv.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        series[label] = load_loop_lengths(path)

    n = len(series)
    fig, axes = plt.subplots(1, n + 1, figsize=(4.2 * (n + 1), 3.8), sharey=True)
    if n + 1 == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n, 1)))
    all_lengths = np.concatenate(list(series.values())) if series else np.array([0])
    bin_edges = np.linspace(max(0, all_lengths.min() - 1), all_lengths.max() + 1, bins + 1)

    # Overlay
    ax0 = axes[0]
    for (label, lengths), color in zip(series.items(), colors):
        ax0.hist(
            lengths,
            bins=bin_edges,
            alpha=0.45,
            label=f"{label} (n={len(lengths)})",
            color=color,
            edgecolor="white",
            linewidth=0.3,
        )
    ax0.set_title("Overlay")
    ax0.set_xlabel("DFG–APE loop length")
    ax0.set_ylabel("Count")
    ax0.legend(fontsize=7)

    for ax, (label, lengths), color in zip(axes[1:], series.items(), colors):
        ax.hist(lengths, bins=bin_edges, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(f"{label}\nn={len(lengths)}")
        ax.set_xlabel("DFG–APE loop length")
        med = float(np.median(lengths)) if len(lengths) else float("nan")
        ax.axvline(med, color="black", linestyle="--", linewidth=1, label=f"median={med:.0f}")
        ax.legend(fontsize=7)

    fig.suptitle("Activation-loop length by filter method", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved histogram → {save_path}")
    return save_path


def write_dfg_ape_msa_html(
    loop_tsv: str,
    out_html: str,
    *,
    title: str = "DFG–APE loops",
    category: str = "passed",
    aligned_seqs: Optional[Dict[str, str]] = None,
    full_seqs: Optional[Dict[str, str]] = None,
    full_col_labels: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
) -> str:
    """
    Write a Workflow1-style two-tab HTML MSA for one method's pass set.

    DFG–APE panel: left-aligned / padded loops from ``loop_tsv``, or
    ``aligned_seqs`` when provided (e.g. MUSCLE column-trimmed gapped loops).
    Full panel: ``full_seqs`` if given, else the same DFG–APE sequences.
    """
    df = pd.read_csv(loop_tsv, sep="\t")
    if "pdb_basename" not in df.columns or "loop_sequence" not in df.columns:
        raise KeyError(f"{loop_tsv} must have pdb_basename and loop_sequence")

    names = [str(n) for n in df["pdb_basename"].tolist()]
    if max_rows is not None and len(names) > max_rows:
        names = names[:max_rows]

    if aligned_seqs is not None:
        dfg_ape_seqs = {n: aligned_seqs.get(n, "") for n in names}
        max_len = max((len(s) for s in dfg_ape_seqs.values()), default=0)
        dfg_ape_seqs = {n: s.ljust(max_len, "-") for n, s in dfg_ape_seqs.items()}
        dfg_labels = [str(i + 1) for i in range(max_len)]
    else:
        loop_map = {
            str(r.pdb_basename): str(r.loop_sequence)
            for r in df.itertuples(index=False)
        }
        seqs = {n: loop_map.get(n, "") for n in names}
        max_len = max((len(s) for s in seqs.values()), default=0)
        dfg_ape_seqs = {n: s.ljust(max_len, "-") for n, s in seqs.items()}
        dfg_labels = [str(i + 1) for i in range(max_len)]

    if full_seqs is None:
        full_seqs = dfg_ape_seqs
        full_col_labels = dfg_labels
    else:
        full_seqs = {n: full_seqs.get(n, "") for n in names}
        if full_col_labels is None:
            n_cols = max((len(s) for s in full_seqs.values()), default=0)
            full_col_labels = [str(i + 1) for i in range(n_cols)]

    row_meta = {n: (category, "") for n in names}
    _write_html_file(
        out_html,
        title,
        names,
        full_seqs,
        full_col_labels,
        dfg_ape_seqs,
        dfg_labels,
        row_meta,
    )
    return out_html


def muscle_column_aligned_loops(
    msa_dict: Dict[str, str],
    passing_names: Sequence[str],
    dfg_cols: Sequence[int],
    ape_cols: Sequence[int],
) -> Dict[str, str]:
    """Return gapped DFG→APE loop subsequences from a MUSCLE MSA (between motifs)."""
    loop_start = int(dfg_cols[-1]) + 1
    loop_end = int(ape_cols[0])
    out = {}
    for name in passing_names:
        if name not in msa_dict:
            continue
        out[name] = msa_dict[name][loop_start:loop_end]
    return out


def summarize_method_counts(method_to_loop_tsv: Dict[str, str]) -> pd.DataFrame:
    """Return a small summary table of n_passing and length stats per method."""
    rows = []
    for label, path in method_to_loop_tsv.items():
        lengths = load_loop_lengths(path)
        rows.append(
            {
                "method": label,
                "n_passing": int(len(lengths)),
                "length_min": int(lengths.min()) if len(lengths) else None,
                "length_median": float(np.median(lengths)) if len(lengths) else None,
                "length_max": int(lengths.max()) if len(lengths) else None,
                "loop_tsv": path,
            }
        )
    return pd.DataFrame(rows)


def copy_basenames_to_dir(
    source_dir: str,
    basenames: Iterable[str],
    dest_dir: str,
) -> int:
    """Copy ``basename.pdb`` files from ``source_dir`` into ``dest_dir``."""
    import shutil

    os.makedirs(dest_dir, exist_ok=True)
    n = 0
    for name in basenames:
        stem = os.path.splitext(os.path.basename(str(name)))[0]
        src = os.path.join(source_dir, f"{stem}.pdb")
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dest_dir, f"{stem}.pdb"))
            n += 1
    return n
