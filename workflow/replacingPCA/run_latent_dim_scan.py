#!/usr/bin/env python3
"""Latent-dimension scan: CNN2d / Small / wr2DCNN (writheCH2) on BRAF CG loops.

Trains and evaluates models across latent dimensions
``{4, 6, 10, 25, 36, 54, 64, 81, 120}`` with 3 repeats each, writing results under
``latentDScan/{d}/{model}/repeat_{r}/``. Builds comparison summary CSVs under
``latentDScan/comparison/``.

Example::

    python run_latent_dim_scan.py
    python run_latent_dim_scan.py --models writheCH2_ae --skip-existing
    python run_latent_dim_scan.py --latent-dim 4 6 --repeats 1
    python run_latent_dim_scan.py --summary-only
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

DEFAULT_DATA_DIR = os.path.join(
    _REPO_ROOT, "Results", "activation_segments", "fitted"
)
DEFAULT_BASE_OUTPUT_DIR = os.path.join(_REPO_ROOT, "Results", "cnn2d_fitted")
DEFAULT_LATENT_SCAN_ROOT = "latentDScan"
DEFAULT_SCAN_DIMS = [4, 6, 10, 25, 36, 54, 64, 81, 120]
DEFAULT_MODELS = ["cnn2d_ae", "small_ae", "writheCH2_ae"]
DEFAULT_WRITHECH2_BETA = 0.0275

MODEL_CONFIG = {
    "cnn2d_ae": {"short": "CNN2d"},
    "small_ae": {"short": "Small"},
    "writheCH2_ae": {"short": "wr2DCNN"},
}

COMPARISON_METRIC_COLS = [
    "best_valid_loss",
    "best_valid_epoch",
    "mean_rmsd_train",
    "mean_rmsd_valid",
    "bondlength_r_train",
    "bondlength_r_valid",
    "angle_r_train",
    "angle_r_valid",
]


def run_subfolder(
    latent_dim: int,
    model: str,
    repeat: int,
    root: str = DEFAULT_LATENT_SCAN_ROOT,
) -> str:
    return f"{root}/{latent_dim}/{model}/repeat_{repeat}"


def comparison_dir(args: argparse.Namespace) -> str:
    return os.path.join(args.base_output_dir, args.latent_scan_root, "comparison")


def _has_checkpoint(run_dir: str) -> bool:
    return bool(glob.glob(os.path.join(run_dir, "checkpoint_*.ckpt")))


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_training_log_metrics(run_dir: str) -> dict:
    """Best valid loss / epoch from molearn log if present."""
    row: dict = {}
    for log_name in ("log.dat", "log_file.dat"):
        log_path = os.path.join(run_dir, "xbb_foldingnet_checkpoints", log_name)
        if not os.path.isfile(log_path):
            continue
        try:
            log_df = pd.read_csv(log_path)
        except Exception as exc:
            print(f"  WARNING: could not read {log_path}: {exc}")
            continue
        if "valid_loss" not in log_df.columns:
            continue
        best_idx = log_df["valid_loss"].idxmin()
        row["best_valid_loss"] = float(log_df.loc[best_idx, "valid_loss"])
        if "epoch" in log_df.columns:
            row["best_valid_epoch"] = int(log_df.loc[best_idx, "epoch"])
        break
    return row


def extract_run_metrics(
    run_dir: str,
    latent_dim: int,
    model: str,
    repeat: int,
    subfolder: str,
) -> dict:
    """Read scalar metrics from a completed run directory."""
    from .ae_aligned_export import _pearson_r

    row: dict = {
        "latent_dim": latent_dim,
        "model": model,
        "repeat": repeat,
        "subfolder": subfolder,
    }
    if not os.path.isdir(run_dir):
        return row

    row.update(_read_training_log_metrics(run_dir))

    rmsd_csv = os.path.join(run_dir, "rmsd_per_frame_train_valid.csv")
    if os.path.isfile(rmsd_csv):
        rmsd_df = pd.read_csv(rmsd_csv)
        if "split" in rmsd_df.columns and "rmsd_angstrom" in rmsd_df.columns:
            for split in ("train", "valid"):
                mask = rmsd_df["split"] == split
                if mask.any():
                    row[f"mean_rmsd_{split}"] = float(
                        rmsd_df.loc[mask, "rmsd_angstrom"].mean()
                    )

    for split in ("train", "valid"):
        bl_csv = os.path.join(run_dir, f"ca_bondlength_per_bond_{split}.csv")
        if os.path.isfile(bl_csv):
            bl_df = pd.read_csv(bl_csv)
            row[f"bondlength_r_{split}"] = float(
                _pearson_r(
                    bl_df["mean_input_angstrom"].values,
                    bl_df["mean_decoded_angstrom"].values,
                )
            )
        ang_csv = os.path.join(run_dir, f"ca_angle_per_residue_{split}.csv")
        if os.path.isfile(ang_csv):
            ang_df = pd.read_csv(ang_csv)
            row[f"angle_r_{split}"] = float(
                _pearson_r(
                    ang_df["mean_input_rad"].values,
                    ang_df["mean_decoded_rad"].values,
                )
            )

    return row


def train_model(wf, model: str, latent_dim: int, subfolder: str, args) -> None:
    if model == "cnn2d_ae":
        wf.train_cnn2d_ae(
            max_epochs=args.max_epochs,
            patience=args.patience,
            latent_dim=latent_dim,
            init_c=args.init_c,
            m=args.m,
            min_size=args.min_size,
            output_subfolder=subfolder,
        )
    elif model == "small_ae":
        wf.train_small_ae(
            max_epochs=args.max_epochs,
            patience=args.patience,
            latent_dimension=latent_dim,
            output_subfolder=subfolder,
        )
    elif model == "writheCH2_ae":
        wf.train_writheCH2_ae(
            max_epochs=args.max_epochs,
            patience=args.patience,
            latent_dim=latent_dim,
            init_c=args.init_c,
            m=args.m,
            min_size=args.min_size,
            beta=args.beta,
            output_subfolder=subfolder,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def run_analysis(wf, subfolder: str, atom_selection) -> None:
    if wf.net is None:
        wf.load_checkpoint()
    wf.setup_analysis(atom_selection=atom_selection)
    _clear_cuda_cache()
    aligned = wf.export_kabsch_aligned_datasets()
    wf.plot_rmsd_comparison(aligned_export=aligned, subfolder=subfolder, show=False)
    wf.plot_rg_comparison(aligned_export=aligned, subfolder=subfolder, show=False)
    wf.plot_rmsf_comparison(aligned_export=aligned, subfolder=subfolder, show=False)
    wf.plot_ca_bondlength_comparison(
        aligned_export=aligned, subfolder=subfolder, show=False
    )
    wf.plot_ca_angle_comparison(
        aligned_export=aligned, subfolder=subfolder, show=False
    )
    _clear_cuda_cache()


def run_pipeline_for_run(wf, latent_dim: int, model: str, repeat: int, args) -> dict | None:
    """Full train + analysis pipeline for one (latent_dim, model, repeat)."""
    subfolder = run_subfolder(latent_dim, model, repeat, args.latent_scan_root)
    run_dir = os.path.join(args.base_output_dir, subfolder)

    print(
        f"\n{'=' * 60}\n"
        f"{model} latent_dim={latent_dim} repeat={repeat} -> {subfolder}\n"
        f"{'=' * 60}"
    )

    wf.output_root_dir = args.base_output_dir

    skip_train = args.summary_only or (
        args.skip_existing and _has_checkpoint(run_dir)
    )
    if skip_train:
        if args.summary_only:
            print("  --summary-only: skipping training")
        else:
            print(f"  --skip-existing: checkpoint found in {run_dir}")
        wf.set_output_subfolder(subfolder, create=False)
    else:
        train_model(wf, model, latent_dim, subfolder, args)
        wf.set_output_subfolder(subfolder, create=False)

    if args.train_only:
        print("  --train-only: skipping analysis")
        return None

    if not os.path.isdir(run_dir):
        print(f"  SKIP analysis: directory missing ({run_dir})")
        return None

    if not skip_train or not args.summary_only:
        run_analysis(wf, subfolder, args.atom_selection)
    else:
        # Re-export / replot from existing checkpoint when summary-only.
        if wf.net is None:
            wf.load_checkpoint()
        if getattr(wf, "data_train", None) is None:
            wf.setup_analysis(atom_selection=args.atom_selection)
        aligned = wf.export_kabsch_aligned_datasets()
        wf.plot_rmsd_comparison(aligned_export=aligned, subfolder=subfolder, show=False)
        wf.plot_rg_comparison(aligned_export=aligned, subfolder=subfolder, show=False)
        wf.plot_rmsf_comparison(aligned_export=aligned, subfolder=subfolder, show=False)
        wf.plot_ca_bondlength_comparison(
            aligned_export=aligned, subfolder=subfolder, show=False
        )
        wf.plot_ca_angle_comparison(
            aligned_export=aligned, subfolder=subfolder, show=False
        )

    return extract_run_metrics(run_dir, latent_dim, model, repeat, subfolder)


def _aggregate_model_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in COMPARISON_METRIC_COLS if c in summary_df.columns]
    if not numeric_cols:
        return pd.DataFrame()

    grouped = summary_df.groupby(["latent_dim", "model"], as_index=False)
    mean_df = grouped[numeric_cols].mean()
    std_df = grouped[numeric_cols].std()
    mean_df = mean_df.rename(columns={c: f"{c}_mean" for c in numeric_cols})
    std_df = std_df.rename(columns={c: f"{c}_std" for c in numeric_cols})
    return mean_df.merge(std_df, on=["latent_dim", "model"])


def _build_wide_comparison(agg_df: pd.DataFrame) -> pd.DataFrame:
    if agg_df.empty:
        return pd.DataFrame()

    wide_rows = []
    for latent_dim, group in agg_df.groupby("latent_dim"):
        row: dict = {"latent_dim": int(latent_dim)}
        for model in sorted(group["model"].unique()):
            model_row = group[group["model"] == model].iloc[0]
            short = MODEL_CONFIG.get(model, {}).get("short", model)
            for metric in COMPARISON_METRIC_COLS:
                mean_col = f"{metric}_mean"
                std_col = f"{metric}_std"
                if mean_col in model_row.index and pd.notna(model_row[mean_col]):
                    row[f"{short}_{metric}_mean"] = model_row[mean_col]
                if std_col in model_row.index and pd.notna(model_row[std_col]):
                    row[f"{short}_{metric}_std"] = model_row[std_col]
        wide_rows.append(row)

    return pd.DataFrame(wide_rows).sort_values("latent_dim").reset_index(drop=True)


def build_comparison_artifacts(
    summary_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Write comparison summary CSVs under latentDScan/comparison/."""
    comp_root = comparison_dir(args)
    per_dim_dir = os.path.join(comp_root, "per_dim")
    os.makedirs(per_dim_dir, exist_ok=True)

    summary_path = os.path.join(comp_root, "latent_dim_scan_summary.csv")
    summary_df = summary_df.sort_values(
        ["latent_dim", "model", "repeat"]
    ).reset_index(drop=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved run summary -> {summary_path}")

    agg_df = _aggregate_model_stats(summary_df)
    if agg_df.empty:
        print("No numeric metrics found; skipping aggregated comparison tables.")
        return summary_df

    agg_path = os.path.join(comp_root, "latent_dim_scan_agg_mean_std.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved mean/std aggregation -> {agg_path}")

    wide_df = _build_wide_comparison(agg_df)
    wide_path = os.path.join(comp_root, "model_comparison_by_latent_dim.csv")
    wide_df.to_csv(wide_path, index=False)
    print(f"Saved wide comparison -> {wide_path}")

    for latent_dim, dim_group in agg_df.groupby("latent_dim"):
        dim_csv = os.path.join(
            per_dim_dir, f"d{int(latent_dim)}_model_comparison_summary.csv"
        )
        dim_group.to_csv(dim_csv, index=False)

    print(f"Saved per-dim comparison tables -> {per_dim_dir}")
    return summary_df


def collect_summary_rows(args: argparse.Namespace) -> list[dict]:
    """Gather metrics from all expected (dim, model, repeat) combinations."""
    rows: list[dict] = []
    dims = args.summary_dims if args.summary_dims is not None else args.latent_dims

    for latent_dim in dims:
        for model in args.models:
            for repeat in range(1, args.repeats + 1):
                subfolder = run_subfolder(
                    latent_dim, model, repeat, args.latent_scan_root
                )
                run_dir = os.path.join(args.base_output_dir, subfolder)
                print(
                    f"\n=== summary: {model} d={latent_dim} repeat={repeat} "
                    f"({run_dir}) ==="
                )
                if not os.path.isdir(run_dir):
                    print("  SKIP: directory missing")
                    continue
                rows.append(
                    extract_run_metrics(
                        run_dir, latent_dim, model, repeat, subfolder
                    )
                )
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Latent-dimension scan: cnn2d_ae vs small_ae "
            "(train + Kabsch metrics + comparison)."
        )
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        nargs="+",
        default=None,
        metavar="D",
        help=f"Latent dimensions to train/analyze (default: {DEFAULT_SCAN_DIMS})",
    )
    parser.add_argument(
        "--summary-dims",
        type=int,
        nargs="+",
        default=None,
        metavar="D",
        help="Latent dimensions for summary/comparison (default: same as --latent-dim)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        choices=list(MODEL_CONFIG.keys()),
        help=f"Models to run (default: {DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per (model, latent_dim) (default: 3)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip training; rebuild comparison artifacts from existing runs",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training when checkpoint_*.ckpt already exists in the run folder",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train only; skip decode/analysis for each run",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not build latentDScan/comparison/ artifacts at the end",
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-output-dir", default=DEFAULT_BASE_OUTPUT_DIR)
    parser.add_argument("--latent-scan-root", default=DEFAULT_LATENT_SCAN_ROOT)
    parser.add_argument("--max-epochs", type=int, default=32)
    parser.add_argument("--patience", type=int, default=32)
    parser.add_argument("--init-c", type=int, default=32)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--min-size", type=int, default=9)
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_WRITHECH2_BETA,
        help=(
            f"writheCH2 reciprocal-DM loss weight "
            f"(default: {DEFAULT_WRITHECH2_BETA}; used only for writheCH2_ae)"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=25,
        help="Base seed; repeat r uses seed + (r-1)",
    )
    parser.add_argument(
        "--atom-selection",
        nargs="+",
        default=["CA"],
        help="Atom names for training/analysis (default: CA)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="CUDA device (e.g. cuda:1). Default: auto freest GPU.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from .autoencoder_workflow import AutoencoderWorkflow

    args.atom_selection = list(args.atom_selection)
    args.latent_dims = (
        args.latent_dim if args.latent_dim is not None else list(DEFAULT_SCAN_DIMS)
    )
    args.models = args.models if args.models is not None else list(DEFAULT_MODELS)
    if args.summary_dims is None:
        args.summary_dims = list(args.latent_dims)

    args.base_output_dir = os.path.abspath(args.base_output_dir)
    args.data_dir = os.path.abspath(args.data_dir)
    os.makedirs(args.base_output_dir, exist_ok=True)

    print(f"Data dir: {args.data_dir}")
    print(f"Latent dims: {args.latent_dims}")
    print(f"Summary dims: {args.summary_dims}")
    print(f"Models: {args.models}")
    print(f"Repeats: {args.repeats}")
    print(
        f"Output root: {os.path.join(args.base_output_dir, args.latent_scan_root)}"
    )

    summary_rows: list[dict] = []

    if args.summary_only:
        summary_rows = collect_summary_rows(args)
    else:
        for latent_dim in args.latent_dims:
            for repeat in range(1, args.repeats + 1):
                seed = args.seed + (repeat - 1)
                for model in args.models:
                    wf = AutoencoderWorkflow(
                        folder_name=args.data_dir,
                        output_base_dir=args.base_output_dir,
                        manual_seed=seed,
                        batch_size=args.batch_size,
                        device=args.device,
                    )
                    wf.prepare_data(atom_selection=args.atom_selection)
                    row = run_pipeline_for_run(
                        wf, latent_dim, model, repeat, args
                    )
                    if row is not None:
                        summary_rows.append(row)
                    _clear_cuda_cache()

        if not args.no_summary and not args.train_only:
            summary_rows = collect_summary_rows(args)

    if not args.no_summary and not args.train_only:
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            build_comparison_artifacts(summary_df, args)
        else:
            print("\nNo summary rows collected; skipping comparison artifacts.")

    print("\nLatent dimension scan complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
