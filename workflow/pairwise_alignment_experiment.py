"""Pairwise FoldMason vs MUSTANG comparison (template + one PDB loop).

Used by 08c-PairwiseAlignmentExperiment.ipynb. Metrics match the multi-MSA experiment panels:
runtime, coverage (mean ± SEM across pairs), conservation overlap at ≥70%.
"""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from workflow.structural_alignment_experiment import (
    alignment_coverage_stats,
    conserved_residue_set,
)


DEFAULT_PDB_DIR = "Results/activation_segments/misaligned_filter/"
DEFAULT_REFERENCE_PDB = "6UAN_chainD.pdb"
DEFAULT_EXPERIMENT_DIR = "Results/Experiments/pairwise_alignment"
DEFAULT_MUSTANG_BIN = "/home/marmatt/Downloads/MUSTANG_v3.2.4/bin/mustang-3.2.4"
DEFAULT_THRESHOLD = 0.70


def _mean_sem(values: Sequence[float]) -> tuple:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return mean, sem, n


def collect_coverage_rows(
    fasta_paths: Sequence[str],
    *,
    method: str,
) -> pd.DataFrame:
    rows = []
    for path in fasta_paths:
        stats = alignment_coverage_stats(path)
        rows.append(
            {
                "method": method,
                "fasta": path,
                "structure": os.path.basename(os.path.dirname(path)),
                "align_len": stats["align_len"],
                "fully_aligned_cols": stats["fully_aligned_cols"],
                "mean_gap_fraction": stats["mean_gap_fraction"],
                "n_sequences": stats["n_sequences"],
            }
        )
    return pd.DataFrame(rows)


def _conserved_df_from_alignments(
    alignments: list,
    *,
    reference_residues: list,
    conservation_threshold: float,
    output_plot: str,
    output_csv: str,
    show_plot: bool = False,
) -> pd.DataFrame:
    from workflow.analyse_alignment_foldmason import analyse_alignment

    if not alignments:
        empty = pd.DataFrame(columns=["position", "residue", "conservation"])
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        empty.to_csv(output_csv, index=False)
        return empty

    analyser = analyse_alignment()
    conservation, _ = analyser.visualize_residue_conservation(
        filtered_alignments=alignments,
        reference_residues=reference_residues,
        output_file=output_plot,
        show_plot=show_plot,
    )
    if conservation is None:
        empty = pd.DataFrame(columns=["position", "residue", "conservation"])
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        empty.to_csv(output_csv, index=False)
        return empty

    conserved_indices = np.where(conservation >= conservation_threshold)[0]
    conserved_df = pd.DataFrame(
        {
            "position": conserved_indices,
            "residue": [
                reference_residues[i]
                if reference_residues is not None and i < len(reference_residues)
                else "N/A"
                for i in conserved_indices
            ],
            "conservation": [float(conservation[i]) for i in conserved_indices],
        }
    )
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    conserved_df.to_csv(output_csv, index=False)
    print(f"Saved conserved residues to: {output_csv} ({len(conserved_df)} residues)")
    return conserved_df


def conservation_from_mustang_dir(
    mustang_dir: str,
    *,
    reference_residues: list,
    conservation_threshold: float = DEFAULT_THRESHOLD,
    output_plot: str,
    output_csv: str,
    show_plot: bool = False,
) -> pd.DataFrame:
    from workflow.analyse_alignment_foldmason import analyse_alignment

    analyser = analyse_alignment(alignment_dir=mustang_dir)
    alignments = analyser.load_alignments(force_reload=True)
    return _conserved_df_from_alignments(
        alignments,
        reference_residues=reference_residues,
        conservation_threshold=conservation_threshold,
        output_plot=output_plot,
        output_csv=output_csv,
        show_plot=show_plot,
    )


def conservation_from_foldmason_pair_fastas(
    fasta_paths: Sequence[str],
    *,
    reference_residues: list,
    conservation_threshold: float = DEFAULT_THRESHOLD,
    output_plot: str,
    output_csv: str,
    show_plot: bool = False,
) -> pd.DataFrame:
    from workflow.analyse_alignment_foldmason import analyse_alignment

    alignments = []
    for path in fasta_paths:
        name = os.path.basename(os.path.dirname(path))
        tmp = analyse_alignment()
        seq1, seq2 = tmp.afasta_parse(path)
        if seq1 is None or seq2 is None:
            continue
        alignments.append(analyse_alignment(name, seq1, seq2))
    return _conserved_df_from_alignments(
        alignments,
        reference_residues=reference_residues,
        conservation_threshold=conservation_threshold,
        output_plot=output_plot,
        output_csv=output_csv,
        show_plot=show_plot,
    )


def run_pairwise_comparison(
    *,
    pdb_dir: str = DEFAULT_PDB_DIR,
    template_pdb: str = DEFAULT_REFERENCE_PDB,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    mustang_bin: str = DEFAULT_MUSTANG_BIN,
    conservation_threshold: float = DEFAULT_THRESHOLD,
    show_conservation_plots: bool = False,
    foldmason_report_mode: int = 0,
) -> Dict[str, object]:
    """
    Run full pairwise FoldMason and MUSTANG loops; return metrics dict + DataFrames.
    """
    if not os.path.exists(template_pdb):
        raise FileNotFoundError(f"Reference PDB not found: {template_pdb}")
    if not os.path.isdir(pdb_dir):
        raise FileNotFoundError(
            f"Sample pool directory not found: {pdb_dir}. "
            "Regenerate Results/activation_segments/misaligned_filter/ upstream."
        )

    from workflow.align_FoldMason import AlignmentFoldMason
    from workflow.align_Mustang import AlignmentMustang
    from workflow.utilities import braf_res

    os.makedirs(experiment_dir, exist_ok=True)
    fm_dir = os.path.join(experiment_dir, "foldmason")
    mu_dir = os.path.join(experiment_dir, "mustang")
    os.makedirs(fm_dir, exist_ok=True)
    os.makedirs(mu_dir, exist_ok=True)

    print("=" * 80)
    print("  FoldMason pairwise loop")
    print("=" * 80)
    aligner_fm = AlignmentFoldMason(
        log_file=os.path.join(fm_dir, "foldmason_pairwise.log")
    )
    fm_summary = aligner_fm.process_foldmason_alignment(
        pdb_path=pdb_dir,
        target_dir=fm_dir,
        template_pdb=template_pdb,
        report_mode=foldmason_report_mode,
    )
    print(
        f"FoldMason done in {fm_summary['elapsed_s']:.1f} s "
        f"({fm_summary['successful']}/{fm_summary['n_input']} ok)"
    )

    print("=" * 80)
    print("  MUSTANG pairwise loop")
    print("=" * 80)
    aligner_mu = AlignmentMustang(
        mustang_path=mustang_bin,
        log_file=os.path.join(mu_dir, "mustang_pairwise.log"),
    )
    mu_summary = aligner_mu.process_mustang_alignment(
        pdb_path=pdb_dir,
        target_dir=mu_dir,
        template_pdb=template_pdb,
    )
    print(
        f"MUSTANG done in {mu_summary['elapsed_s']:.1f} s "
        f"({mu_summary['successful']}/{mu_summary['n_input']} ok)"
    )

    fm_cov = collect_coverage_rows(fm_summary["fasta_paths"], method="FoldMason")
    mu_cov = collect_coverage_rows(mu_summary["afasta_paths"], method="MUSTANG")
    coverage_df = pd.concat([fm_cov, mu_cov], ignore_index=True)
    coverage_csv = os.path.join(experiment_dir, "per_pair_coverage.csv")
    coverage_df.to_csv(coverage_csv, index=False)
    print(f"Saved per-pair coverage: {coverage_csv}")

    reference_residues = braf_res()
    conserved_fm_df = conservation_from_foldmason_pair_fastas(
        fm_summary["fasta_paths"],
        reference_residues=reference_residues,
        conservation_threshold=conservation_threshold,
        output_plot=os.path.join(fm_dir, "conservation_foldmason.png"),
        output_csv=os.path.join(fm_dir, "conserved_residues_70pct_foldmason.csv"),
        show_plot=show_conservation_plots,
    )
    conserved_mu_df = conservation_from_mustang_dir(
        mu_dir,
        reference_residues=reference_residues,
        conservation_threshold=conservation_threshold,
        output_plot=os.path.join(mu_dir, "conservation_mustang.png"),
        output_csv=os.path.join(mu_dir, "conserved_residues_70pct_mustang.csv"),
        show_plot=show_conservation_plots,
    )

    res_fm = conserved_residue_set(conserved_fm_df)
    res_mu = conserved_residue_set(conserved_mu_df)
    shared = res_fm & res_mu
    only_fm = res_fm - res_mu
    only_mu = res_mu - res_fm
    union = res_fm | res_mu
    jaccard = (len(shared) / len(union)) if union else float("nan")

    summary = {
        "n_input": int(fm_summary["n_input"]),
        "t_foldmason_s": float(fm_summary["elapsed_s"]),
        "t_mustang_s": float(mu_summary["elapsed_s"]),
        "t_foldmason_min": float(fm_summary["elapsed_s"]) / 60.0,
        "t_mustang_min": float(mu_summary["elapsed_s"]) / 60.0,
        "fm_successful": int(fm_summary["successful"]),
        "mu_successful": int(mu_summary["successful"]),
        "fm_failed": int(fm_summary["failed"]),
        "mu_failed": int(mu_summary["failed"]),
        "n_conserved_fm": len(res_fm),
        "n_conserved_mu": len(res_mu),
        "n_only_fm": len(only_fm),
        "n_both": len(shared),
        "n_only_mu": len(only_mu),
        "jaccard": jaccard,
        "conservation_threshold": conservation_threshold,
    }
    summary_csv = os.path.join(experiment_dir, "summary_metrics.csv")
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    with open(os.path.join(experiment_dir, "summary_metrics.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary: {summary_csv}")

    return {
        "summary": summary,
        "coverage_df": coverage_df,
        "conserved_fm_df": conserved_fm_df,
        "conserved_mu_df": conserved_mu_df,
        "experiment_dir": experiment_dir,
    }


def plot_runtime_comparison(
    summary: Dict[str, object],
    *,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    show: bool = True,
) -> pd.DataFrame:
    methods = ["FoldMason", "MUSTANG"]
    means = [float(summary["t_foldmason_min"]), float(summary["t_mustang_min"])]
    out = pd.DataFrame(
        {
            "Method": methods,
            "Runtime_min": means,
            "Runtime_s": [float(summary["t_foldmason_s"]), float(summary["t_mustang_s"])],
            "successful": [summary["fm_successful"], summary["mu_successful"]],
            "failed": [summary["fm_failed"], summary["mu_failed"]],
        }
    )
    print(out.to_string(index=False))
    out_csv = os.path.join(experiment_dir, "runtime_comparison.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(methods, means, color=["steelblue", "darkorange"], edgecolor="white")
    ax.set_ylabel("Runtime (min)")
    n = summary.get("n_input", "?")
    ax.set_title(f"Pairwise alignment runtime — {n} structures")
    plt.tight_layout()
    out_png = os.path.join(experiment_dir, "runtime_comparison.png")
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot: {out_png}")
    return out


def plot_coverage_comparison(
    coverage_df: pd.DataFrame,
    *,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    show: bool = True,
) -> pd.DataFrame:
    metric_specs = [
        ("Alignment length", "align_len"),
        ("Fully aligned columns", "fully_aligned_cols"),
        ("Mean gap fraction", "mean_gap_fraction"),
    ]
    methods = ["FoldMason", "MUSTANG"]
    rows = []
    for label, col in metric_specs:
        for method in methods:
            sub = coverage_df.loc[coverage_df["method"] == method, col]
            mean, sem, n = _mean_sem(sub.tolist())
            rows.append({"Metric": label, "Method": method, "mean": mean, "sem": sem, "n": n})
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    out_csv = os.path.join(experiment_dir, "coverage_comparison.csv")
    summary.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    colors = ["steelblue", "darkorange"]
    x = np.arange(len(methods))
    for ax, (label, _) in zip(axes, metric_specs):
        sub = summary[summary["Metric"] == label].set_index("Method")
        means = [float(sub.loc[m, "mean"]) if m in sub.index else 0.0 for m in methods]
        sems = [float(sub.loc[m, "sem"]) if m in sub.index else 0.0 for m in methods]
        ax.bar(x, means, yerr=sems, color=colors, capsize=4, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(label)
        ax.set_ylabel("Value")
    n_pairs = int(coverage_df.groupby("method").size().max()) if len(coverage_df) else 0
    fig.suptitle(f"Pairwise coverage (mean ± SEM across pairs, n≈{n_pairs})", y=1.05)
    plt.tight_layout()
    out_png = os.path.join(experiment_dir, "coverage_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot: {out_png}")
    return summary


def plot_conservation_overlap(
    summary: Dict[str, object],
    *,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    show: bool = True,
) -> pd.DataFrame:
    labels = ["FoldMason only", "Both", "MUSTANG only"]
    counts = [
        int(summary["n_only_fm"]),
        int(summary["n_both"]),
        int(summary["n_only_mu"]),
    ]
    jaccard = float(summary["jaccard"])
    out = pd.DataFrame(
        {
            "Category": labels + ["Jaccard"],
            "value": counts + [jaccard],
        }
    )
    print(out.to_string(index=False))
    print(f"\nJaccard similarity: {jaccard:.3f}")
    out_csv = os.path.join(experiment_dir, "conservation_overlap.csv")
    out.to_csv(out_csv, index=False)
    print(f"Saved summary: {out_csv}")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(
        labels,
        counts,
        color=["steelblue", "mediumseagreen", "darkorange"],
        edgecolor="white",
    )
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Number of residues")
    thr = float(summary.get("conservation_threshold", 0.70)) * 100
    ax.set_title(f"Conservation overlap (≥{thr:.0f} %, pairwise)")
    plt.tight_layout()
    out_png = os.path.join(experiment_dir, "conservation_overlap.png")
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot: {out_png}")
    return out
