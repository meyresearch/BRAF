"""
Per-frame writhe chirality scalars and KDE comparison plots for autoencoder models.

Decomposes the Klenin-Langowski writhe matrix into positive and negative parts
per frame and compares input vs encode->decode distributions.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import gaussian_kde

from .wrCNN2D import AutoEncoder as WrAutoEncoder

if TYPE_CHECKING:
    from .autoencoder_workflow import AutoencoderWorkflow


def writhe_pos_neg_batch(coords, device=None, batch_size=256):
    """Return per-frame (pos_wr, neg_wr) from Klenin-Langowski writhe matrices."""
    coords_t = torch.as_tensor(np.asarray(coords), dtype=torch.float32)
    if device is None:
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    pos_list, neg_list = [], []
    with torch.no_grad():
        for start in range(0, len(coords_t), batch_size):
            batch = coords_t[start : start + batch_size].to(device)
            sigma = WrAutoEncoder.coords_to_writhe(batch).squeeze(1)
            pos_list.append(sigma.clamp(min=0).sum(dim=(-2, -1)).cpu().numpy())
            neg_list.append(sigma.clamp(max=0).sum(dim=(-2, -1)).cpu().numpy())
    return np.concatenate(pos_list), np.concatenate(neg_list)


def compute_writhe_chirality_per_frame(
    wf: "AutoencoderWorkflow",
    *,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    subfolder: Optional[str] = None,
    writhe_batch_size: int = 256,
    csv_filename: str = "writhe_chirality_per_frame.csv",
):
    """
    Compute per-frame positive/negative writhe sums for input and decoded structures.

    Uses Kabsch-aligned real-Å coordinates from ``aligned_export`` or
    ``aligned_coordinates.npz`` (writhe is invariant under the global
    ``(x-mean)/std`` scale and under rigid alignment).
    """
    from .ae_aligned_export import _get_coords_for_plots, _wf_subfolder

    subfolder = _wf_subfolder(wf, subfolder)
    train_in, valid_in, train_out, valid_out = _get_coords_for_plots(
        wf,
        aligned_export,
        npz_path,
        subfolder=subfolder,
    )
    device = getattr(wf, "device", None)

    splits = {}
    rows = []
    for name, cin, cout in (
        ("train", train_in, train_out),
        ("valid", valid_in, valid_out),
    ):
        pos_in, neg_in = writhe_pos_neg_batch(
            cin, device=device, batch_size=writhe_batch_size
        )
        pos_dec, neg_dec = writhe_pos_neg_batch(
            cout, device=device, batch_size=writhe_batch_size
        )
        splits[name] = {
            "pos_in": pos_in,
            "neg_in": neg_in,
            "pos_dec": pos_dec,
            "neg_dec": neg_dec,
        }
        for i in range(len(pos_in)):
            rows.append(
                {
                    "split": name,
                    "pos_wr_input": float(pos_in[i]),
                    "neg_wr_input": float(neg_in[i]),
                    "pos_wr_decoded": float(pos_dec[i]),
                    "neg_wr_decoded": float(neg_dec[i]),
                }
            )

    csv_path = os.path.join(wf.output_base_dir, csv_filename)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

    pos_in_all = np.concatenate([splits["train"]["pos_in"], splits["valid"]["pos_in"]])
    neg_in_all = np.concatenate([splits["train"]["neg_in"], splits["valid"]["neg_in"]])
    pos_dec_all = np.concatenate(
        [splits["train"]["pos_dec"], splits["valid"]["pos_dec"]]
    )
    neg_dec_all = np.concatenate(
        [splits["train"]["neg_dec"], splits["valid"]["neg_dec"]]
    )

    return {
        "csv": csv_path,
        "splits": splits,
        "pos_in": pos_in_all,
        "neg_in": neg_in_all,
        "pos_dec": pos_dec_all,
        "neg_dec": neg_dec_all,
    }


def _finite(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return vals[np.isfinite(vals)]


def _plot_kde_panel(ax, vals_in, vals_dec):
    """Draw input/decoded KDEs with mean vlines; skip KDE if variance is ~0."""
    series = [
        (_finite(vals_in), "input", "tab:blue", "-"),
        (_finite(vals_dec), "decoded", "tab:orange", "--"),
    ]
    pooled = (
        np.concatenate([v for v, _, _, _ in series if v.size])
        if any(v.size for v, _, _, _ in series)
        else np.array([0.0])
    )
    x_lo, x_hi = float(pooled.min()), float(pooled.max())
    if x_lo == x_hi:
        pad = 0.05 if x_lo == 0 else abs(x_lo) * 0.05
        x_lo, x_hi = x_lo - pad, x_hi + pad
    else:
        span = x_hi - x_lo
        x_lo -= 0.05 * span
        x_hi += 0.05 * span
    x_range = np.linspace(x_lo, x_hi, 400)

    for vals, lbl, color, ls in series:
        if vals.size == 0:
            continue
        ax.axvline(
            vals.mean(),
            color=color,
            linestyle=":",
            alpha=0.8,
            label=f"{lbl} mean = {vals.mean():.3f}",
        )
        if vals.size < 2 or np.allclose(vals, vals[0]):
            continue
        kde = gaussian_kde(vals)
        ax.fill_between(x_range, kde(x_range), alpha=0.25, color=color)
        ax.plot(x_range, kde(x_range), color=color, linestyle=ls, lw=1.8, label=lbl)


def plot_writhe_chirality_distribution(
    wf: "AutoencoderWorkflow",
    *,
    results: Optional[dict] = None,
    aligned_export: Optional[Mapping[str, np.ndarray]] = None,
    npz_path: Optional[str] = None,
    subfolder: Optional[str] = None,
    model_label: Optional[str] = None,
    figure_filename: str = "writhe_chirality_distribution.png",
    dpi: int = 200,
    show: bool = True,
):
    """Two-panel KDE: positive writhe sum (left) and negative writhe sum (right)."""
    if results is None:
        results = compute_writhe_chirality_per_frame(
            wf,
            aligned_export=aligned_export,
            npz_path=npz_path,
            subfolder=subfolder,
        )

    pos_in = results["pos_in"]
    neg_in = results["neg_in"]
    pos_dec = results["pos_dec"]
    neg_dec = results["neg_dec"]

    if model_label is None:
        model_label = (
            wf.network_class.__name__
            if hasattr(wf.network_class, "__name__")
            else str(wf.network_class)
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    panels = (
        (
            axes[0],
            pos_in,
            pos_dec,
            r"Positive writhe sum $\sum_{i,j}\max(\Sigma_{ij},0)$",
        ),
        (
            axes[1],
            neg_in,
            neg_dec,
            r"Negative writhe sum $\sum_{i,j}\min(\Sigma_{ij},0)$",
        ),
    )

    for ax, vals_in, vals_dec, xlabel in panels:
        _plot_kde_panel(ax, vals_in, vals_dec)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Writhe chirality: input vs decoded  ({model_label})")
    fig.tight_layout()
    png_path = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    print(f"Saved {png_path}")

    return {"figure": png_path, "results": results}
