"""FoldMason vs MUSTANG structural-alignment experiment (multi-seed).

Ported from workflowMarch2026/Experiments/StructuralAlignment.ipynb for use in
08b-MultiMSAAlignmentExperiment.ipynb. Runs repeated random subsamples and aggregates
comparison metrics as mean ± SEM.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from glob import glob
from time import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_N_SAMPLE = 50
DEFAULT_MUSTANG_BIN = "/home/marmatt/Downloads/MUSTANG_v3.2.4/bin/mustang-3.2.4"
DEFAULT_EXPERIMENT_DIR = "Results/Experiments/structural_alignment"
DEFAULT_RECONSTR_DIR = "Results/activation_segments/misaligned_filter/"
DEFAULT_REFERENCE_PDB = "6UAN_chainD.pdb"


def parse_fasta(filepath: str) -> Dict[str, str]:
    """Parse a (possibly multi-sequence) FASTA or AFASTA file into a dict."""
    sequences: Dict[str, str] = {}
    current_name = None
    current_seq: List[str] = []
    with open(filepath) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if current_name is not None:
                    sequences[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0].split(".pdb")[0]
                current_seq = []
            elif line:
                current_seq.append(line)
    if current_name is not None:
        sequences[current_name] = "".join(current_seq)
    return sequences


def sample_pdbs(
    reconstr_dir: str,
    n_sample: int = DEFAULT_N_SAMPLE,
    seed: int = 42,
    exclude_ref_basename: str = "6UAN_chainD",
) -> List[str]:
    """Randomly sample *n_sample* PDB paths from *reconstr_dir* (no replacement)."""
    if not os.path.isdir(reconstr_dir):
        raise FileNotFoundError(
            f"Sample pool directory not found: {reconstr_dir}. "
            "Regenerate Results/activation_segments/misaligned_filter/ upstream."
        )
    all_pdbs = sorted(glob(os.path.join(reconstr_dir, "*.pdb")))
    all_pdbs = [p for p in all_pdbs if exclude_ref_basename not in os.path.basename(p)]
    if not all_pdbs:
        raise FileNotFoundError(
            f"No .pdb files found in: {reconstr_dir}. "
            "Regenerate Results/activation_segments/misaligned_filter/ upstream."
        )
    if n_sample > len(all_pdbs):
        raise ValueError(
            f"Requested N_SAMPLE={n_sample} but only {len(all_pdbs)} PDBs available in {reconstr_dir}."
        )
    random.seed(seed)
    return random.sample(all_pdbs, n_sample)


def stage_inputs(pdb_list: Sequence[str], out_dir: str) -> List[str]:
    """Clear *out_dir* of existing PDBs and copy *pdb_list* into it."""
    os.makedirs(out_dir, exist_ok=True)
    for stale in glob(os.path.join(out_dir, "*.pdb")):
        os.remove(stale)
    staged = []
    for pdb in pdb_list:
        dest = os.path.join(out_dir, os.path.basename(pdb))
        shutil.copy2(pdb, dest)
        staged.append(dest)
    return staged


def alignment_coverage_stats(fasta_path: str) -> Dict[str, object]:
    """Compute alignment length / fully-aligned columns / mean gap fraction."""
    seqs = parse_fasta(fasta_path)
    if not seqs:
        return {
            "n_sequences": 0,
            "align_len": 0,
            "fully_aligned_cols": 0,
            "mean_gap_fraction": float("nan"),
            "gap_fractions": [],
        }
    gap_fractions = [s.count("-") / len(s) if len(s) > 0 else 0.0 for s in seqs.values()]
    align_len = len(next(iter(seqs.values())))
    seq_array = np.array([list(s) for s in seqs.values()])
    fully_aligned_cols = int(np.sum(np.all(seq_array != "-", axis=0)))
    return {
        "n_sequences": len(seqs),
        "align_len": align_len,
        "fully_aligned_cols": fully_aligned_cols,
        "mean_gap_fraction": float(np.mean(gap_fractions)),
        "gap_fractions": gap_fractions,
    }


def conserved_residue_set(conserved_df: pd.DataFrame) -> Set[str]:
    if conserved_df is None or len(conserved_df) == 0:
        return set()
    if "residue" in conserved_df.columns:
        return set(conserved_df["residue"].astype(str).tolist())
    return set(conserved_df.iloc[:, 0].astype(str).tolist())


def run_foldmason_timed(
    *,
    input_dir: str,
    output_dir: str,
    template_pdb: str,
    out_name: str = "msa",
    report_mode: int = 2,
    log_file: Optional[str] = None,
) -> Tuple[float, Optional[str]]:
    from workflow.align_FoldMason import AlignmentFoldMason

    os.makedirs(output_dir, exist_ok=True)
    if log_file is None:
        log_file = os.path.join(output_dir, "foldmason_msa.log")
    aligner = AlignmentFoldMason(log_file=log_file)
    t0 = time()
    out_prefix = aligner.process_foldmason_alignment_multi(
        pdb_path=input_dir,
        target_dir=output_dir,
        template_pdb=os.path.abspath(template_pdb),
        out_name=out_name,
        report_mode=report_mode,
    )
    elapsed = time() - t0
    return elapsed, out_prefix


def run_mustang_timed(
    *,
    input_dir: str,
    output_prefix: str,
    mustang_bin: str = DEFAULT_MUSTANG_BIN,
) -> Tuple[float, str]:
    if not os.path.exists(mustang_bin):
        raise FileNotFoundError(f"MUSTANG binary not found: {mustang_bin}")
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)
    mustang_cmd = (
        f"{mustang_bin} "
        f"-i {input_dir}/*.pdb "
        f"-o {output_prefix} "
        f"-F fasta "
        f"-s ON"
    )
    print("Running MUSTANG MSA...")
    print(f"Command: {mustang_cmd}\n")
    t0 = time()
    result = subprocess.run(mustang_cmd, shell=True, capture_output=True, text=True)
    elapsed = time() - t0
    if result.returncode != 0:
        print("MUSTANG stderr:")
        print(result.stderr)
        raise RuntimeError("MUSTANG exited with non-zero return code.")
    if result.stdout:
        print(result.stdout)
    afasta = output_prefix + ".afasta"
    if not os.path.exists(afasta):
        raise FileNotFoundError(f"MUSTANG did not produce expected AFASTA: {afasta}")
    return elapsed, afasta


def run_conservation(
    *,
    alignment_file: str,
    reference_name: str,
    reference_residues: list,
    output_plot: str,
    output_csv: str,
    conservation_threshold: float = 0.70,
    show_plot: bool = False,
) -> pd.DataFrame:
    from workflow.analyse_alignment_foldmason import analyse_alignment

    analyser = analyse_alignment()
    result = analyser.run_multi_alignment_conservation_analysis(
        alignment_file=alignment_file,
        reference_name=reference_name,
        reference_residues=reference_residues,
        conservation_threshold=conservation_threshold,
        output_plot=output_plot,
        output_csv=output_csv,
        show_plot=show_plot,
        verbose=True,
    )
    return result["conserved_df"]


def run_one_seed(
    seed: int,
    *,
    reconstr_dir: str = DEFAULT_RECONSTR_DIR,
    reference_pdb: str = DEFAULT_REFERENCE_PDB,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    n_sample: int = DEFAULT_N_SAMPLE,
    mustang_bin: str = DEFAULT_MUSTANG_BIN,
    conservation_threshold: float = 0.70,
    show_conservation_plots: bool = False,
) -> Dict[str, object]:
    """Run FoldMason + MUSTANG for one seed; return a metrics row dict."""
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")

    seed_dir = os.path.join(experiment_dir, f"seed_{seed}")
    fm_input = os.path.join(seed_dir, "foldmason", "input")
    fm_output = os.path.join(seed_dir, "foldmason")
    mu_input = os.path.join(seed_dir, "mustang", "input")
    mu_output = os.path.join(seed_dir, "mustang")
    os.makedirs(seed_dir, exist_ok=True)

    sampled = sample_pdbs(reconstr_dir, n_sample=n_sample, seed=seed)
    all_input = [os.path.abspath(p) for p in sampled] + [os.path.abspath(reference_pdb)]
    print(f"\n{'=' * 80}\n  Seed {seed}: {len(sampled)} sampled + 1 reference\n{'=' * 80}")

    stage_inputs(all_input, fm_input)
    stage_inputs(all_input, mu_input)

    t_fm, fm_prefix = run_foldmason_timed(
        input_dir=fm_input,
        output_dir=fm_output,
        template_pdb=reference_pdb,
        out_name="msa",
        report_mode=2,
    )
    if not fm_prefix:
        raise RuntimeError(f"FoldMason failed for seed {seed}")
    print(f"FoldMason MSA completed in {t_fm:.1f} s ({t_fm / 60:.1f} min)")

    mu_prefix = os.path.join(mu_output, "mustang_msa")
    t_mu, mu_afasta = run_mustang_timed(
        input_dir=mu_input,
        output_prefix=mu_prefix,
        mustang_bin=mustang_bin,
    )
    print(f"MUSTANG MSA completed in {t_mu:.1f} s ({t_mu / 60:.1f} min)")

    fm_3di = os.path.join(fm_output, "msa_3di.fa")
    if not os.path.exists(fm_3di):
        raise FileNotFoundError(f"FoldMason 3Di FASTA missing: {fm_3di}")

    fm_cov = alignment_coverage_stats(fm_3di)
    mu_cov = alignment_coverage_stats(mu_afasta)

    from workflow.utilities import braf_res

    # Use default BRAF reference residue list (matches production notebooks).
    reference_residues = braf_res()
    reference_name = os.path.splitext(os.path.basename(reference_pdb))[0]

    conserved_fm_df = run_conservation(
        alignment_file=fm_3di,
        reference_name=reference_name,
        reference_residues=reference_residues,
        output_plot=os.path.join(fm_output, "conservation_foldmason.png"),
        output_csv=os.path.join(fm_output, "conserved_residues_70pct_foldmason.csv"),
        conservation_threshold=conservation_threshold,
        show_plot=show_conservation_plots,
    )
    conserved_mu_df = run_conservation(
        alignment_file=mu_afasta,
        reference_name=reference_name,
        reference_residues=reference_residues,
        output_plot=os.path.join(mu_output, "conservation_mustang.png"),
        output_csv=os.path.join(mu_output, "conserved_residues_70pct_mustang.csv"),
        conservation_threshold=conservation_threshold,
        show_plot=show_conservation_plots,
    )

    res_fm = conserved_residue_set(conserved_fm_df)
    res_mu = conserved_residue_set(conserved_mu_df)
    shared = res_fm & res_mu
    only_fm = res_fm - res_mu
    only_mu = res_mu - res_fm
    union = res_fm | res_mu
    jaccard = (len(shared) / len(union)) if union else float("nan")

    # Persist sample list for reproducibility
    with open(os.path.join(seed_dir, "sampled_pdbs.txt"), "w") as fh:
        for p in sampled:
            fh.write(os.path.basename(p) + "\n")

    return {
        "seed": seed,
        "n_sample": n_sample,
        "n_input": len(all_input),
        "t_foldmason_s": float(t_fm),
        "t_mustang_s": float(t_mu),
        "t_foldmason_min": float(t_fm) / 60.0,
        "t_mustang_min": float(t_mu) / 60.0,
        "fm_align_len": fm_cov["align_len"],
        "mu_align_len": mu_cov["align_len"],
        "fm_fully_aligned_cols": fm_cov["fully_aligned_cols"],
        "mu_fully_aligned_cols": mu_cov["fully_aligned_cols"],
        "fm_mean_gap_fraction": fm_cov["mean_gap_fraction"],
        "mu_mean_gap_fraction": mu_cov["mean_gap_fraction"],
        "n_conserved_fm": len(res_fm),
        "n_conserved_mu": len(res_mu),
        "n_only_fm": len(only_fm),
        "n_both": len(shared),
        "n_only_mu": len(only_mu),
        "jaccard": jaccard,
        "seed_dir": seed_dir,
        "fm_3di_fa": fm_3di,
        "mu_afasta": mu_afasta,
    }


def run_all_seeds(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    *,
    reconstr_dir: str = DEFAULT_RECONSTR_DIR,
    reference_pdb: str = DEFAULT_REFERENCE_PDB,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    n_sample: int = DEFAULT_N_SAMPLE,
    mustang_bin: str = DEFAULT_MUSTANG_BIN,
    conservation_threshold: float = 0.70,
    show_conservation_plots: bool = False,
) -> pd.DataFrame:
    """Run the experiment for each seed and write ``per_seed_metrics.csv``."""
    os.makedirs(experiment_dir, exist_ok=True)
    rows = []
    for seed in seeds:
        rows.append(
            run_one_seed(
                int(seed),
                reconstr_dir=reconstr_dir,
                reference_pdb=reference_pdb,
                experiment_dir=experiment_dir,
                n_sample=n_sample,
                mustang_bin=mustang_bin,
                conservation_threshold=conservation_threshold,
                show_conservation_plots=show_conservation_plots,
            )
        )
    df = pd.DataFrame(rows)
    out_csv = os.path.join(experiment_dir, "per_seed_metrics.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-seed metrics: {out_csv}")
    return df


def aggregate_mean_sem(
    df: pd.DataFrame,
    value_cols: Iterable[str],
) -> pd.DataFrame:
    """Return mean / SEM over seed repeats for the given numeric columns."""
    cols = list(value_cols)
    n = len(df)
    rows = []
    for col in cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        mean = float(vals.mean())
        if n > 1:
            sem = float(vals.std(ddof=1) / np.sqrt(n))
        else:
            sem = float("nan")
        rows.append({"metric": col, "mean": mean, "sem": sem, "n": n})
    return pd.DataFrame(rows)


def _save_summary_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved summary: {path}")


def plot_runtime_comparison(
    per_seed_df: pd.DataFrame,
    *,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    show: bool = True,
) -> pd.DataFrame:
    """Bar chart: FoldMason vs MUSTANG runtime (min), mean ± SEM."""
    agg = aggregate_mean_sem(per_seed_df, ["t_foldmason_min", "t_mustang_min"])
    methods = ["FoldMason", "MUSTANG"]
    means = [
        float(agg.loc[agg["metric"] == "t_foldmason_min", "mean"].iloc[0]),
        float(agg.loc[agg["metric"] == "t_mustang_min", "mean"].iloc[0]),
    ]
    sems = [
        float(agg.loc[agg["metric"] == "t_foldmason_min", "sem"].iloc[0]),
        float(agg.loc[agg["metric"] == "t_mustang_min", "sem"].iloc[0]),
    ]

    summary = pd.DataFrame(
        {
            "Method": methods,
            "Runtime_min_mean": means,
            "Runtime_min_sem": sems,
            "Runtime_s_mean": [m * 60 for m in means],
            "Runtime_s_sem": [s * 60 for s in sems],
        }
    )
    print(summary.to_string(index=False))
    _save_summary_csv(summary, os.path.join(experiment_dir, "runtime_comparison.csv"))

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(
        methods,
        means,
        yerr=sems,
        color=["steelblue", "darkorange"],
        capsize=4,
        edgecolor="white",
    )
    ax.set_ylabel("Runtime (min)")
    n = len(per_seed_df)
    n_structs = int(per_seed_df["n_input"].iloc[0]) if "n_input" in per_seed_df.columns else "?"
    ax.set_title(f"Alignment runtime — {n_structs} structures\n(mean ± SEM, n={n})")
    plt.tight_layout()
    out_png = os.path.join(experiment_dir, "runtime_comparison.png")
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot: {out_png}")
    return summary


def plot_coverage_comparison(
    per_seed_df: pd.DataFrame,
    *,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    show: bool = True,
) -> pd.DataFrame:
    """Grouped bars for alignment length, fully aligned columns, mean gap fraction."""
    metric_specs = [
        ("Alignment length", "fm_align_len", "mu_align_len"),
        ("Fully aligned columns", "fm_fully_aligned_cols", "mu_fully_aligned_cols"),
        ("Mean gap fraction", "fm_mean_gap_fraction", "mu_mean_gap_fraction"),
    ]
    rows = []
    for label, fm_col, mu_col in metric_specs:
        for method, col in [("FoldMason", fm_col), ("MUSTANG", mu_col)]:
            vals = pd.to_numeric(per_seed_df[col], errors="coerce")
            n = len(vals)
            mean = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            rows.append(
                {
                    "Metric": label,
                    "Method": method,
                    "mean": mean,
                    "sem": sem,
                    "n": n,
                }
            )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    _save_summary_csv(summary, os.path.join(experiment_dir, "coverage_comparison.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    methods = ["FoldMason", "MUSTANG"]
    colors = ["steelblue", "darkorange"]
    x = np.arange(len(methods))

    for ax, label in zip(axes, [m[0] for m in metric_specs]):
        sub = summary[summary["Metric"] == label].set_index("Method")
        means = [float(sub.loc[m, "mean"]) for m in methods]
        sems = [float(sub.loc[m, "sem"]) for m in methods]
        ax.bar(x, means, yerr=sems, color=colors, capsize=4, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(label)
        ax.set_ylabel(label.split()[-1] if "fraction" in label.lower() else "Count")

    n = len(per_seed_df)
    fig.suptitle(f"Alignment coverage (mean ± SEM, n={n})", y=1.05)
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
    per_seed_df: pd.DataFrame,
    *,
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
    show: bool = True,
) -> pd.DataFrame:
    """Venn-style bars: FoldMason only / Both / MUSTANG only (mean ± SEM)."""
    agg = aggregate_mean_sem(per_seed_df, ["n_only_fm", "n_both", "n_only_mu", "jaccard"])
    labels = ["FoldMason only", "Both", "MUSTANG only"]
    keys = ["n_only_fm", "n_both", "n_only_mu"]
    means = [float(agg.loc[agg["metric"] == k, "mean"].iloc[0]) for k in keys]
    sems = [float(agg.loc[agg["metric"] == k, "sem"].iloc[0]) for k in keys]
    j_mean = float(agg.loc[agg["metric"] == "jaccard", "mean"].iloc[0])
    j_sem = float(agg.loc[agg["metric"] == "jaccard", "sem"].iloc[0])

    summary = pd.DataFrame(
        {
            "Category": labels + ["Jaccard"],
            "mean": means + [j_mean],
            "sem": sems + [j_sem],
            "n": [len(per_seed_df)] * 4,
        }
    )
    print(summary.to_string(index=False))
    print(f"\nJaccard similarity (mean ± SEM): {j_mean:.3f} ± {j_sem:.3f}")
    _save_summary_csv(summary, os.path.join(experiment_dir, "conservation_overlap.csv"))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(
        labels,
        means,
        yerr=sems,
        color=["steelblue", "mediumseagreen", "darkorange"],
        edgecolor="white",
        capsize=4,
    )
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (sem if np.isfinite(sem) else 0) + 0.3,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Number of residues")
    n = len(per_seed_df)
    ax.set_title(f"Conservation overlap (≥70 %)\n(mean ± SEM, n={n})")
    plt.tight_layout()
    out_png = os.path.join(experiment_dir, "conservation_overlap.png")
    plt.savefig(out_png, dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved plot: {out_png}")
    return summary
