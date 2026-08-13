"""
Kabsch-aligned decoded coordinates, multi-MODEL CA PDBs, NumPy archive,
and per-frame RMSD tables for ``AutoencoderWorkflow`` models.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from .autoencoder_workflow import AutoencoderWorkflow


def compute_rmsd_no_align(coords_in, coords_out):
    """Per-frame RMSD without Kabsch alignment.

    ``coords_in``, ``coords_out``: real-Å arrays ``[n_frames, n_atoms, 3]``.
    """
    sq_diff = (coords_in - coords_out) ** 2
    return np.sqrt(sq_diff.sum(axis=-1).mean(axis=-1))


def decode_to_real_coords(net, coords_std, mean, std, batch_size=64, n_atoms=None):
    """Encode + decode standardised coords; return real-Å ``float32`` numpy.

    Small_AutoEncoder decoders emit slightly more points than ``out_points``
    (e.g. 80 for 74 CAs). Molearn's trainer trims with ``[:, :batch.size(1), :]``;
    we apply the same slice here so exports match the input atom count.
    """
    if n_atoms is None:
        n_atoms = int(coords_std.shape[1])
    device = next(net.parameters()).device
    net.eval()
    out_chunks = []
    with torch.no_grad():
        for start in range(0, len(coords_std), batch_size):
            batch = coords_std[start : start + batch_size].to(device).float()
            z = net.encode(batch)
            decoded = net.decode(z)[:, :n_atoms, :]
            out_chunks.append(decoded.cpu().numpy())
    decoded_std = np.concatenate(out_chunks, axis=0)
    return (decoded_std * float(std) + float(mean)).astype(np.float32)


def kabsch_align(P, Q):
    """Align *P* (query) onto *Q* (reference): rotation + translation, det(R)=+1."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    cP = P.mean(axis=0)
    cQ = Q.mean(axis=0)
    H = (P - cP).T @ (Q - cQ)
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    if d == 0:
        d = 1.0
    R = V @ np.diag([1.0, 1.0, d]) @ U.T
    return ((P - cP) @ R.T + cQ).astype(np.float32)


def kabsch_align_batch(decoded_raw, input_coords):
    """Per-frame Kabsch: align each decoded frame onto its input."""
    return np.stack(
        [
            kabsch_align(decoded_raw[i], input_coords[i])
            for i in range(len(input_coords))
        ],
        axis=0,
    )


def write_multimodel_ca_pdb(coords_array, path, chain="A", resname="ALA"):
    """Write a multi-MODEL CA-only PDB: ``coords_array`` ``[n_frames, n_atoms, 3]`` Å."""
    coords_array = np.asarray(coords_array, dtype=np.float32)
    assert coords_array.ndim == 3 and coords_array.shape[2] == 3, (
        f"expected [n_frames, n_atoms, 3], got {coords_array.shape}"
    )
    with open(path, "w") as f:
        for model_idx, coords in enumerate(coords_array, start=1):
            f.write(f"MODEL    {model_idx:5d}\n")
            for i, (x, y, z) in enumerate(coords, start=1):
                f.write(
                    f"ATOM  {i:5d}  CA  {resname} {chain}{i:4d}    "
                    f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
                    f"  1.00  0.00           C\n"
                )
            f.write("ENDMDL\n")
        f.write("END\n")


def _summarise(name, arr):
    print(
        f"{name:>11s}: n={len(arr):4d}  "
        f"mean={arr.mean():7.3f} Å  median={np.median(arr):7.3f} Å  "
        f"min={arr.min():7.3f} Å  max={arr.max():7.3f} Å"
    )


def radius_of_gyration_per_frame(coords):
    """Per-frame radius of gyration (Å). ``coords``: ``[n_frames, n_atoms, 3]``."""
    coords = np.asarray(coords, dtype=np.float64)
    com = coords.mean(axis=1, keepdims=True)
    sq_dist = ((coords - com) ** 2).sum(axis=-1)
    return np.sqrt(sq_dist.mean(axis=-1)).astype(np.float32)


def per_ca_rmsf(coords):
    """Per-residue RMSF (Å) over frames. ``coords``: ``[n_frames, n_atoms, 3]``."""
    coords = np.asarray(coords, dtype=np.float64)
    mean_pos = coords.mean(axis=0)
    sq_disp = ((coords - mean_pos) ** 2).sum(axis=-1)
    return np.sqrt(sq_disp.mean(axis=0)).astype(np.float32)


def consecutive_ca_bond_lengths(coords):
    """Consecutive Cα–Cα distances (Å). ``coords``: ``[n_frames, n_atoms, 3]``."""
    coords = np.asarray(coords, dtype=np.float64)
    diff = coords[:, 1:, :] - coords[:, :-1, :]
    return np.linalg.norm(diff, axis=-1).astype(np.float32)


def consecutive_ca_angles(coords):
    """Cα–Cα–Cα angles (rad) at each middle residue. ``coords``: ``[n_frames, n_atoms, 3]``."""
    coords = np.asarray(coords, dtype=np.float64)
    p0 = coords[:, :-2, :]
    p1 = coords[:, 1:-1, :]
    p2 = coords[:, 2:, :]
    b0 = p0 - p1
    b1 = p2 - p1
    b0_norm = b0 / np.linalg.norm(b0, axis=-1, keepdims=True)
    b1_norm = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    dot_product = np.einsum("...i,...i->...", b0_norm, b1_norm)
    return np.arccos(np.clip(dot_product, -1.0, 1.0)).astype(np.float32)


def _pearson_r(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _wf_subfolder(wf: "AutoencoderWorkflow", subfolder: Optional[str] = None) -> str:
    """Infer workflow output subfolder for aligned-coordinate archive lookup."""
    return subfolder or getattr(wf, "output_subfolder", None) or "wrCNN2D_ae"


def resolve_aligned_npz_path(
    wf: "AutoencoderWorkflow",
    npz_path: Optional[str] = None,
    *,
    subfolder: Optional[str] = None,
) -> str:
    """Locate ``aligned_coordinates.npz`` under the workflow output tree."""
    subfolder = _wf_subfolder(wf, subfolder)
    if npz_path is not None and os.path.isfile(npz_path):
        return npz_path

    rel = os.path.join("aligned_pdbs", "aligned_coordinates.npz")
    root = getattr(wf, "output_root_dir", None) or wf.output_base_dir
    candidates = []
    if wf.output_base_dir:
        candidates.append(os.path.join(wf.output_base_dir, rel))
    if root:
        if root != wf.output_base_dir:
            candidates.append(os.path.join(root, rel))
        if subfolder:
            candidates.append(os.path.join(root, subfolder, rel))

    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if os.path.isfile(path):
            return path

    tried = "\n  ".join(seen) if seen else "(no candidates)"
    raise FileNotFoundError(
        "Coordinate archive not found. Tried:\n  "
        f"{tried}\n"
        "Run export_kabsch_aligned_datasets() after "
        "wf.set_output_subfolder(...) (or pass npz_path=...)."
    )


def load_input_decoded_coords(
    wf: "AutoencoderWorkflow",
    *,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    subfolder: Optional[str] = None,
):
    """
    Return ``(train_in, valid_in, train_decoded_aligned, valid_decoded_aligned)`` in Å.

    Loads from ``aligned_export``, then ``aligned_coordinates.npz``, without
    requiring ``wf.data_train`` to be populated. Decoded coordinates are
    Kabsch-aligned onto input per frame.
    """
    return _load_coords_from_export_or_npz(
        wf, aligned_export, npz_path, subfolder=_wf_subfolder(wf, subfolder)
    )


def _decoded_aligned_from_mapping(mapping: Mapping[str, np.ndarray]):
    """Resolve Kabsch-aligned decoded arrays from an export dict or NPZ mapping."""
    in_keys = ("train_in_real", "valid_in_real")
    aligned_keys = ("train_decoded_aligned", "valid_decoded_aligned")
    raw_keys = ("train_decoded_raw", "valid_decoded_raw")
    if any(k not in mapping for k in in_keys):
        return None
    train_in, valid_in = mapping[in_keys[0]], mapping[in_keys[1]]
    if all(k in mapping for k in aligned_keys):
        return train_in, valid_in, mapping[aligned_keys[0]], mapping[aligned_keys[1]]
    if all(k in mapping for k in raw_keys):
        train_raw, valid_raw = mapping[raw_keys[0]], mapping[raw_keys[1]]
        return (
            train_in,
            valid_in,
            kabsch_align_batch(train_raw, train_in),
            kabsch_align_batch(valid_raw, valid_in),
        )
    return None


def _load_coords_from_export_or_npz(
    wf: "AutoencoderWorkflow",
    aligned_export: Optional[Mapping[str, np.ndarray]],
    npz_path: Optional[str],
    *,
    subfolder: Optional[str] = None,
):
    """Return train/valid input and Kabsch-aligned decoded arrays."""
    subfolder = _wf_subfolder(wf, subfolder)
    if aligned_export is not None:
        result = _decoded_aligned_from_mapping(aligned_export)
        if result is not None:
            return result

    npz_path = resolve_aligned_npz_path(wf, npz_path, subfolder=subfolder)
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(
            f"Coordinate archive not found: {npz_path}. "
            "Run export_kabsch_aligned_datasets() first."
        )
    data = np.load(npz_path)
    result = _decoded_aligned_from_mapping(data)
    if result is not None:
        return result
    raise KeyError(
        f"{npz_path} must contain train/valid input plus either "
        "train_decoded_aligned/valid_decoded_aligned or train_decoded_raw/valid_decoded_raw."
    )


def _get_coords_for_plots(
    wf: "AutoencoderWorkflow",
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    *,
    train_in: Optional[np.ndarray] = None,
    valid_in: Optional[np.ndarray] = None,
    train_out: Optional[np.ndarray] = None,
    valid_out: Optional[np.ndarray] = None,
    subfolder: Optional[str] = None,
):
    """Return train/valid input and Kabsch-aligned decoded coordinate arrays."""
    if train_in is not None and valid_in is not None and train_out is not None and valid_out is not None:
        return train_in, valid_in, train_out, valid_out
    return _load_coords_from_export_or_npz(
        wf, aligned_export, npz_path, subfolder=_wf_subfolder(wf, subfolder)
    )


def plot_rmsd_comparison(
    wf: "AutoencoderWorkflow",
    *,
    train_in: Optional[np.ndarray] = None,
    valid_in: Optional[np.ndarray] = None,
    train_out: Optional[np.ndarray] = None,
    valid_out: Optional[np.ndarray] = None,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    csv_filename: str = "rmsd_kabsch_aligned_per_frame.csv",
    figure_filename: str = "rmsd_kabsch_aligned_hist.png",
    dpi: int = 200,
    show: bool = True,
    subfolder: Optional[str] = None,
):
    """Per-frame RMSD histograms (train/valid) for input vs Kabsch-aligned decoded."""
    subfolder = _wf_subfolder(wf, subfolder)
    train_in, valid_in, train_out, valid_out = _get_coords_for_plots(
        wf,
        aligned_export,
        npz_path,
        train_in=train_in,
        valid_in=valid_in,
        train_out=train_out,
        valid_out=valid_out,
        subfolder=subfolder,
    )
    print(
        f"Loaded coordinates: train {train_in.shape}, valid {valid_in.shape}"
    )

    rmsd_train = compute_rmsd_no_align(train_in, train_out)
    rmsd_valid = compute_rmsd_no_align(valid_in, valid_out)

    print("Per-frame RMSD (Kabsch-aligned decoded onto input):")
    _summarise("train", rmsd_train)
    _summarise("validation", rmsd_valid)

    rmsd_csv = os.path.join(wf.output_base_dir, csv_filename)
    pd.DataFrame(
        {
            "split": ["train"] * len(rmsd_train) + ["valid"] * len(rmsd_valid),
            "frame_local_idx": list(range(len(rmsd_train)))
            + list(range(len(rmsd_valid))),
            "rmsd_kabsch_aligned_angstrom": np.concatenate([rmsd_train, rmsd_valid]),
        }
    ).to_csv(rmsd_csv, index=False)
    print(f"Saved {rmsd_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    hi = max(rmsd_train.max(), rmsd_valid.max()) * 1.05
    bins = np.linspace(0, hi, 40)

    axes[0].hist(
        rmsd_train, bins=bins, color="tab:blue", alpha=0.85, edgecolor="black"
    )
    axes[0].axvline(
        rmsd_train.mean(),
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"mean = {rmsd_train.mean():.2f} Å",
    )
    axes[0].axvline(
        np.median(rmsd_train),
        color="black",
        linestyle=":",
        alpha=0.7,
        label=f"median = {np.median(rmsd_train):.2f} Å",
    )
    axes[0].set_xlabel("per-frame RMSD (Kabsch-aligned) [Å]")
    axes[0].set_ylabel("# frames")
    axes[0].set_title(f"Training set  (n={len(rmsd_train)})")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(
        rmsd_valid, bins=bins, color="tab:orange", alpha=0.85, edgecolor="black"
    )
    axes[1].axvline(
        rmsd_valid.mean(),
        color="black",
        linestyle="--",
        alpha=0.7,
        label=f"mean = {rmsd_valid.mean():.2f} Å",
    )
    axes[1].axvline(
        np.median(rmsd_valid),
        color="black",
        linestyle=":",
        alpha=0.7,
        label=f"median = {np.median(rmsd_valid):.2f} Å",
    )
    axes[1].set_xlabel("per-frame RMSD (Kabsch-aligned) [Å]")
    axes[1].set_title(f"Validation set  (n={len(rmsd_valid)})")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Input vs Kabsch-aligned decoded structures — per-frame RMSD")
    fig.tight_layout()
    rmsd_png = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(rmsd_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {rmsd_png}")

    return {
        "figure": rmsd_png,
        "csv": rmsd_csv,
        "rmsd_train": rmsd_train,
        "rmsd_valid": rmsd_valid,
    }


def plot_rg_comparison(
    wf: "AutoencoderWorkflow",
    *,
    train_in: Optional[np.ndarray] = None,
    valid_in: Optional[np.ndarray] = None,
    train_out: Optional[np.ndarray] = None,
    valid_out: Optional[np.ndarray] = None,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    figure_filename: str = "rg_input_vs_decoded.png",
    dpi: int = 200,
    show: bool = True,
    subfolder: Optional[str] = None,
):
    """Rg distributions and input-vs-decoded scatter (train/valid, aligned decoded)."""
    subfolder = _wf_subfolder(wf, subfolder)
    train_in, valid_in, train_out, valid_out = _get_coords_for_plots(
        wf,
        aligned_export,
        npz_path,
        train_in=train_in,
        valid_in=valid_in,
        train_out=train_out,
        valid_out=valid_out,
        subfolder=subfolder,
    )
    splits = (("train", train_in, train_out), ("valid", valid_in, valid_out))
    rg_results = {}

    for name, cin, cout in splits:
        rg_in = radius_of_gyration_per_frame(cin)
        rg_out = radius_of_gyration_per_frame(cout)
        rg_results[name] = (rg_in, rg_out)
        rg_csv = os.path.join(wf.output_base_dir, f"rg_per_frame_{name}.csv")
        pd.DataFrame(
            {
                "frame_local_idx": np.arange(len(rg_in)),
                "rg_input_angstrom": rg_in,
                "rg_decoded_angstrom": rg_out,
            }
        ).to_csv(rg_csv, index=False)
        print(f"Saved {rg_csv}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for col, (name, (rg_in, rg_out)) in enumerate(
        [(s[0], rg_results[s[0]]) for s in splits]
    ):
        r_scatter = _pearson_r(rg_in, rg_out)
        hi = max(rg_in.max(), rg_out.max()) * 1.05
        bins = np.linspace(0, hi, 40)

        ax_hist = axes[0, col]
        ax_hist.hist(
            rg_in, bins=bins, color="tab:blue", alpha=0.55, edgecolor="black", label="input"
        )
        ax_hist.hist(
            rg_out,
            bins=bins,
            color="tab:orange",
            alpha=0.55,
            edgecolor="black",
            label="decoded",
        )
        ax_hist.set_xlabel(r"$R_g$ [Å]")
        ax_hist.set_ylabel("# frames")
        ax_hist.set_title(f"{name.capitalize()} — $R_g$ distribution")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        ax_sc = axes[1, col]
        ax_sc.scatter(rg_in, rg_out, s=12, alpha=0.5, c="tab:green", edgecolors="none")
        lim = (0.0, hi)
        ax_sc.plot(lim, lim, "k--", lw=1.2, label="y = x")
        ax_sc.set_xlim(lim)
        ax_sc.set_ylim(lim)
        ax_sc.set_aspect("equal", adjustable="box")
        ax_sc.set_xlabel(r"$R_g$ input [Å]")
        ax_sc.set_ylabel(r"$R_g$ decoded [Å]")
        ax_sc.set_title(f"{name.capitalize()} — $r$ = {r_scatter:.3f}")
        ax_sc.legend(fontsize=9)
        ax_sc.grid(True, alpha=0.3)
        print(f"{name} Rg Pearson r = {r_scatter:.4f}")

    fig.suptitle("Radius of gyration: input vs Kabsch-aligned decoded")
    fig.tight_layout()
    rg_png = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(rg_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {rg_png}")

    return {
        "figure": rg_png,
        "rg_train": rg_results["train"],
        "rg_valid": rg_results["valid"],
    }


def plot_rmsf_comparison(
    wf: "AutoencoderWorkflow",
    *,
    train_in: Optional[np.ndarray] = None,
    valid_in: Optional[np.ndarray] = None,
    train_out: Optional[np.ndarray] = None,
    valid_out: Optional[np.ndarray] = None,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    figure_filename: str = "rmsf_per_ca_input_vs_decoded.png",
    dpi: int = 200,
    show: bool = True,
    subfolder: Optional[str] = None,
):
    """Per-Cα RMSF curves for input vs Kabsch-aligned decoded (train/valid)."""
    subfolder = _wf_subfolder(wf, subfolder)
    train_in, valid_in, train_out, valid_out = _get_coords_for_plots(
        wf,
        aligned_export,
        npz_path,
        train_in=train_in,
        valid_in=valid_in,
        train_out=train_out,
        valid_out=valid_out,
        subfolder=subfolder,
    )
    splits = (("train", train_in, train_out), ("valid", valid_in, valid_out))
    rmsf_results = {}

    for name, cin, cout in splits:
        rmsf_in = per_ca_rmsf(cin)
        rmsf_out = per_ca_rmsf(cout)
        rmsf_results[name] = (rmsf_in, rmsf_out)
        rmsf_csv = os.path.join(wf.output_base_dir, f"rmsf_per_ca_{name}.csv")
        n_res = len(rmsf_in)
        pd.DataFrame(
            {
                "residue_idx": np.arange(1, n_res + 1),
                "rmsf_input_angstrom": rmsf_in,
                "rmsf_decoded_angstrom": rmsf_out,
            }
        ).to_csv(rmsf_csv, index=False)
        print(f"Saved {rmsf_csv}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (name, (rmsf_in, rmsf_out)) in zip(
        axes, [(s[0], rmsf_results[s[0]]) for s in splits]
    ):
        x_res = np.arange(1, len(rmsf_in) + 1)
        r_rmsf = _pearson_r(rmsf_in, rmsf_out)
        ax.plot(x_res, rmsf_in, color="tab:blue", lw=1.5, label="input")
        ax.plot(
            x_res,
            rmsf_out,
            color="tab:orange",
            lw=1.5,
            ls="--",
            label="decoded",
        )
        ax.set_xlabel("Cα residue index")
        ax.set_ylabel("RMSF [Å]")
        ax.set_title(f"{name.capitalize()} — per-Cα RMSF ($r$ = {r_rmsf:.3f})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        print(f"{name} RMSF Pearson r = {r_rmsf:.4f}")

    fig.suptitle("Per-Cα RMSF: input vs Kabsch-aligned decoded")
    fig.tight_layout()
    rmsf_png = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(rmsf_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {rmsf_png}")

    return {
        "figure": rmsf_png,
        "rmsf_train": rmsf_results["train"],
        "rmsf_valid": rmsf_results["valid"],
    }


def plot_ca_bondlength_comparison(
    wf: "AutoencoderWorkflow",
    *,
    train_in: Optional[np.ndarray] = None,
    valid_in: Optional[np.ndarray] = None,
    train_out: Optional[np.ndarray] = None,
    valid_out: Optional[np.ndarray] = None,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    figure_filename: str = "ca_bondlength_input_vs_decoded.png",
    dpi: int = 200,
    show: bool = True,
    subfolder: Optional[str] = None,
):
    """Consecutive Cα–Cα bond lengths: distributions and per-bond means (train/valid)."""
    subfolder = _wf_subfolder(wf, subfolder)
    train_in, valid_in, train_out, valid_out = _get_coords_for_plots(
        wf,
        aligned_export,
        npz_path,
        train_in=train_in,
        valid_in=valid_in,
        train_out=train_out,
        valid_out=valid_out,
        subfolder=subfolder,
    )
    splits = (("train", train_in, train_out), ("valid", valid_in, valid_out))
    bond_results = {}
    csv_paths = {}

    for name, cin, cout in splits:
        bl_in = consecutive_ca_bond_lengths(cin)
        bl_out = consecutive_ca_bond_lengths(cout)
        mean_in = bl_in.mean(axis=0)
        mean_out = bl_out.mean(axis=0)
        bond_results[name] = (bl_in, bl_out, mean_in, mean_out)
        csv_path = os.path.join(wf.output_base_dir, f"ca_bondlength_per_bond_{name}.csv")
        pd.DataFrame(
            {
                "bond_idx": np.arange(1, len(mean_in) + 1),
                "mean_input_angstrom": mean_in,
                "mean_decoded_angstrom": mean_out,
            }
        ).to_csv(csv_path, index=False)
        csv_paths[name] = csv_path
        print(f"Saved {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (name, (_, _, mean_in, mean_out)) in enumerate(
        [(s[0], bond_results[s[0]]) for s in splits]
    ):
        bl_in, bl_out = bond_results[name][0], bond_results[name][1]
        hi = max(bl_in.max(), bl_out.max()) * 1.05
        bins = np.linspace(0, hi, 40)

        ax_hist = axes[0, col]
        ax_hist.hist(
            bl_in.ravel(),
            bins=bins,
            color="tab:blue",
            alpha=0.55,
            edgecolor="black",
            label="input",
        )
        ax_hist.hist(
            bl_out.ravel(),
            bins=bins,
            color="tab:orange",
            alpha=0.55,
            edgecolor="black",
            label="decoded",
        )
        ax_hist.set_xlabel("Cα–Cα bond length [Å]")
        ax_hist.set_ylabel("# values")
        ax_hist.set_title(f"{name.capitalize()} — bond-length distribution")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        ax_curve = axes[1, col]
        x_bond = np.arange(1, len(mean_in) + 1)
        r_bl = _pearson_r(mean_in, mean_out)
        ax_curve.plot(x_bond, mean_in, color="tab:blue", lw=1.5, label="input")
        ax_curve.plot(
            x_bond,
            mean_out,
            color="tab:orange",
            lw=1.5,
            ls="--",
            label="decoded",
        )
        ax_curve.set_xlabel("Bond index (Cα i → i+1)")
        ax_curve.set_ylabel("Mean bond length [Å]")
        ax_curve.set_title(f"{name.capitalize()} — per-bond mean ($r$ = {r_bl:.3f})")
        ax_curve.legend(fontsize=9)
        ax_curve.grid(True, alpha=0.3)
        print(f"{name} CA–CA bond-length Pearson r = {r_bl:.4f}")

    fig.suptitle("Consecutive Cα–Cα bond lengths: input vs Kabsch-aligned decoded")
    fig.tight_layout()
    bond_png = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(bond_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {bond_png}")

    return {
        "figure": bond_png,
        "csv_train": csv_paths["train"],
        "csv_valid": csv_paths["valid"],
        "bond_train": bond_results["train"],
        "bond_valid": bond_results["valid"],
    }


def plot_ca_angle_comparison(
    wf: "AutoencoderWorkflow",
    *,
    train_in: Optional[np.ndarray] = None,
    valid_in: Optional[np.ndarray] = None,
    train_out: Optional[np.ndarray] = None,
    valid_out: Optional[np.ndarray] = None,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    figure_filename: str = "ca_angle_input_vs_decoded.png",
    dpi: int = 200,
    show: bool = True,
    subfolder: Optional[str] = None,
):
    """Cα–Cα–Cα angles: distributions and per-residue means (train/valid)."""
    subfolder = _wf_subfolder(wf, subfolder)
    train_in, valid_in, train_out, valid_out = _get_coords_for_plots(
        wf,
        aligned_export,
        npz_path,
        train_in=train_in,
        valid_in=valid_in,
        train_out=train_out,
        valid_out=valid_out,
        subfolder=subfolder,
    )
    splits = (("train", train_in, train_out), ("valid", valid_in, valid_out))
    angle_results = {}
    csv_paths = {}

    for name, cin, cout in splits:
        ang_in = consecutive_ca_angles(cin)
        ang_out = consecutive_ca_angles(cout)
        mean_in = ang_in.mean(axis=0)
        mean_out = ang_out.mean(axis=0)
        angle_results[name] = (ang_in, ang_out, mean_in, mean_out)
        csv_path = os.path.join(wf.output_base_dir, f"ca_angle_per_residue_{name}.csv")
        pd.DataFrame(
            {
                "residue_idx": np.arange(2, len(mean_in) + 2),
                "mean_input_rad": mean_in,
                "mean_decoded_rad": mean_out,
            }
        ).to_csv(csv_path, index=False)
        csv_paths[name] = csv_path
        print(f"Saved {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, (name, (_, _, mean_in, mean_out)) in enumerate(
        [(s[0], angle_results[s[0]]) for s in splits]
    ):
        ang_in, ang_out = angle_results[name][0], angle_results[name][1]
        bins = np.linspace(0.0, np.pi, 40)

        ax_hist = axes[0, col]
        ax_hist.hist(
            ang_in.ravel(),
            bins=bins,
            color="tab:blue",
            alpha=0.55,
            edgecolor="black",
            label="input",
        )
        ax_hist.hist(
            ang_out.ravel(),
            bins=bins,
            color="tab:orange",
            alpha=0.55,
            edgecolor="black",
            label="decoded",
        )
        ax_hist.set_xlabel("Cα–Cα–Cα angle [rad]")
        ax_hist.set_ylabel("# values")
        ax_hist.set_title(f"{name.capitalize()} — angle distribution")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        ax_curve = axes[1, col]
        x_res = np.arange(2, len(mean_in) + 2)
        r_ang = _pearson_r(mean_in, mean_out)
        ax_curve.plot(x_res, mean_in, color="tab:blue", lw=1.5, label="input")
        ax_curve.plot(
            x_res,
            mean_out,
            color="tab:orange",
            lw=1.5,
            ls="--",
            label="decoded",
        )
        ax_curve.set_xlabel("Middle Cα residue index")
        ax_curve.set_ylabel("Mean angle [rad]")
        ax_curve.set_title(f"{name.capitalize()} — per-residue mean ($r$ = {r_ang:.3f})")
        ax_curve.legend(fontsize=9)
        ax_curve.grid(True, alpha=0.3)
        print(f"{name} Cα–Cα–Cα angle Pearson r = {r_ang:.4f}")

    fig.suptitle("Cα–Cα–Cα angles: input vs Kabsch-aligned decoded")
    fig.tight_layout()
    angle_png = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(angle_png, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {angle_png}")

    return {
        "figure": angle_png,
        "csv_train": csv_paths["train"],
        "csv_valid": csv_paths["valid"],
        "angle_train": angle_results["train"],
        "angle_valid": angle_results["valid"],
    }


def plot_rg_rmsf_comparison(
    wf: "AutoencoderWorkflow",
    *,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    dpi: int = 200,
    show: bool = True,
    subfolder: Optional[str] = None,
):
    """Plot Rg and RMSF comparisons (convenience wrapper)."""
    subfolder = _wf_subfolder(wf, subfolder)
    rg = plot_rg_comparison(
        wf,
        aligned_export=aligned_export,
        npz_path=npz_path,
        dpi=dpi,
        show=show,
        subfolder=subfolder,
    )
    rmsf = plot_rmsf_comparison(
        wf,
        aligned_export=aligned_export,
        npz_path=npz_path,
        dpi=dpi,
        show=show,
        subfolder=subfolder,
    )
    return {"rg": rg, "rmsf": rmsf}


def export_kabsch_aligned_datasets(
    wf: "AutoencoderWorkflow",
    *,
    train_in_real: Optional[np.ndarray] = None,
    valid_in_real: Optional[np.ndarray] = None,
    train_out_real: Optional[np.ndarray] = None,
    valid_out_real: Optional[np.ndarray] = None,
    batch_size: int = 64,
    pdb_subdir: str = "aligned_pdbs",
    npz_filename: str = "aligned_coordinates.npz",
    rmsd_csv_filename: str = "rmsd_per_frame_train_valid.csv",
    print_rmsd_summary: bool = True,
):
    """
    Decode train/valid, Kabsch-align each decoded frame onto its input, then write:

    * Four multi-MODEL CA PDBs under ``{wf.output_base_dir}/{pdb_subdir}/``
    * ``{npz_filename}`` in that same directory (compressed arrays + mean/std)
    * ``{rmsd_csv_filename}`` in ``wf.output_base_dir`` (Kabsch-aligned RMSD)

    If the ``*_real`` arrays are omitted, they are computed from ``wf`` (same
    convention as the original notebook cells).
    """
    mean = float(wf.data.mean)
    std = float(wf.data.std)

    if train_in_real is None:
        train_in_real = (wf.data_train.cpu().numpy() * std + mean).astype(np.float32)
    if valid_in_real is None:
        valid_in_real = (wf.data_valid.cpu().numpy() * std + mean).astype(np.float32)
    if train_out_real is None:
        train_out_real = decode_to_real_coords(
            wf.net, wf.data_train, mean, std, batch_size=batch_size
        )
    if valid_out_real is None:
        valid_out_real = decode_to_real_coords(
            wf.net, wf.data_valid, mean, std, batch_size=batch_size
        )

    train_decoded_aligned = kabsch_align_batch(train_out_real, train_in_real)
    valid_decoded_aligned = kabsch_align_batch(valid_out_real, valid_in_real)

    rmsd_train_aligned = compute_rmsd_no_align(train_in_real, train_decoded_aligned)
    rmsd_valid_aligned = compute_rmsd_no_align(valid_in_real, valid_decoded_aligned)

    if print_rmsd_summary:
        print("Per-frame RMSD (Kabsch-aligned decoded onto input):")
        _summarise("train", rmsd_train_aligned)
        _summarise("validation", rmsd_valid_aligned)

    viz_dir = os.path.join(wf.output_base_dir, pdb_subdir)
    os.makedirs(viz_dir, exist_ok=True)

    pdb_paths = {
        "input_train": os.path.join(viz_dir, "input_train.pdb"),
        "decoded_aligned_train": os.path.join(viz_dir, "decoded_aligned_train.pdb"),
        "input_valid": os.path.join(viz_dir, "input_valid.pdb"),
        "decoded_aligned_valid": os.path.join(viz_dir, "decoded_aligned_valid.pdb"),
    }

    write_multimodel_ca_pdb(train_in_real, pdb_paths["input_train"])
    write_multimodel_ca_pdb(train_decoded_aligned, pdb_paths["decoded_aligned_train"])
    write_multimodel_ca_pdb(valid_in_real, pdb_paths["input_valid"])
    write_multimodel_ca_pdb(valid_decoded_aligned, pdb_paths["decoded_aligned_valid"])

    print()
    print("Saved multi-MODEL CA-only PDBs (load each input/decoded_aligned pair together):")
    for label, p in pdb_paths.items():
        print(f"  {label:24s} -> {p}  ({os.path.getsize(p) / 1024:.1f} kB)")

    aligned_npz = os.path.join(viz_dir, npz_filename)
    np.savez_compressed(
        aligned_npz,
        train_in_real=train_in_real,
        train_decoded_raw=train_out_real,
        train_decoded_aligned=train_decoded_aligned,
        valid_in_real=valid_in_real,
        valid_decoded_raw=valid_out_real,
        valid_decoded_aligned=valid_decoded_aligned,
        mean=mean,
        std=std,
    )
    print(f"Saved numpy archive: {aligned_npz}")

    rmsd_csv = os.path.join(wf.output_base_dir, rmsd_csv_filename)
    pd.DataFrame(
        {
            "split": ["train"] * len(rmsd_train_aligned) + ["valid"] * len(rmsd_valid_aligned),
            "frame_local_idx": list(range(len(rmsd_train_aligned)))
            + list(range(len(rmsd_valid_aligned))),
            "rmsd_angstrom": np.concatenate([rmsd_train_aligned, rmsd_valid_aligned]),
        }
    ).to_csv(rmsd_csv, index=False)
    print(f"Saved per-frame RMSD table: {rmsd_csv}")

    return {
        "viz_dir": viz_dir,
        "pdb_paths": pdb_paths,
        "aligned_npz": aligned_npz,
        "rmsd_csv": rmsd_csv,
        "train_in_real": train_in_real,
        "valid_in_real": valid_in_real,
        "train_decoded_raw": train_out_real,
        "valid_decoded_raw": valid_out_real,
        "train_decoded_aligned": train_decoded_aligned,
        "valid_decoded_aligned": valid_decoded_aligned,
        "rmsd_train": rmsd_train_aligned,
        "rmsd_valid": rmsd_valid_aligned,
        "rmsd_train_aligned": rmsd_train_aligned,
        "rmsd_valid_aligned": rmsd_valid_aligned,
    }
