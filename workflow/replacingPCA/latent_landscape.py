"""
Pairwise latent-space landscapes for autoencoders with more than two latent
dimensions.

molearn's own grid machinery is hard-wired to 2D: ``MolearnAnalysis.setup_grid``
builds ``np.stack(meshgrid, axis=2).reshape(-1, 1, 2)`` and ``_get_bounds`` only
inspects columns 0 and 1. For a d-dimensional latent space this module instead
scans every ``(i, j)`` pair on a 2D grid while holding the remaining ``d - 2``
dimensions at a fixed reference vector (by default the median of the encoded
training set), so each panel is a median slice through the latent space.

The RMSD surface follows molearn's ``scan_error`` definition: decode a grid
point, re-encode, decode again, and measure how far the model moved the
structure. A well-behaved region of latent space maps back onto itself.

Orientation
-----------
Surfaces are built and stored as ``surface[iy, ix]`` for ``(xvals[ix],
yvals[iy])`` and drawn with ``pcolormesh(xvals, yvals, surface)``, which is what
molearn's ``analysis/plot.py`` does. Drawing the same array with
``imshow(surface.T)`` and matplotlib's default ``origin="upper"`` both flips the
y axis and swaps the axes, which puts the scatter overlay on the wrong cells.
"""

from __future__ import annotations

import os
from itertools import combinations
from typing import TYPE_CHECKING, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

if TYPE_CHECKING:
    from .autoencoder_workflow import AutoencoderWorkflow


def _net_device(net):
    return next(net.parameters()).device


def _resolve_std(wf: "AutoencoderWorkflow") -> float:
    """Å-per-standardised-unit scale used to report RMSD in Ångström."""
    std = getattr(getattr(wf, "data", None), "std", None)
    if std is None and getattr(wf, "MA", None) is not None:
        std = getattr(wf.MA, "stdval", None)
    if std is None:
        raise ValueError(
            "Cannot determine the coordinate standard deviation. "
            "Call prepare_data() before scanning the latent space."
        )
    return float(std)


def encode_batched(net, coords_std, batch_size=256) -> np.ndarray:
    """Encode standardised ``[n_frames, n_atoms, 3]`` coords to ``[n_frames, latent]``."""
    device = _net_device(net)
    coords = torch.as_tensor(np.asarray(coords_std), dtype=torch.float32)
    net.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(coords), batch_size):
            batch = coords[start : start + batch_size].to(device)
            chunks.append(net.encode(batch).reshape(batch.shape[0], -1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64)


def decode_batched(net, z, batch_size=256, n_atoms=None) -> torch.Tensor:
    """Decode ``[n_points, latent]`` to standardised ``[n_points, n_atoms, 3]``.

    Small FoldingNet decoders emit a few more points than ``out_points``; the
    same ``[:, :n_atoms, :]`` trim molearn applies is used here.
    """
    device = _net_device(net)
    z_t = torch.as_tensor(np.asarray(z), dtype=torch.float32)
    net.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(z_t), batch_size):
            batch = net.decode(z_t[start : start + batch_size].to(device))
            if n_atoms is not None:
                batch = batch[:, :n_atoms, :]
            chunks.append(batch.cpu())
    return torch.cat(chunks, dim=0)


def encode_datasets(
    wf: "AutoencoderWorkflow",
    *,
    batch_size: int = 256,
    train_csv: str = "latent_encoded_train.csv",
    valid_csv: str = "latent_encoded_valid.csv",
    save: bool = True,
):
    """Encode the train/validation splits and keep **all** latent dimensions.

    Returns a dict with ``z_train`` / ``z_valid`` of shape ``[n_frames, latent]``.
    """
    if getattr(wf, "net", None) is None:
        raise ValueError("No model loaded. Train or load a checkpoint first.")
    if getattr(wf, "data_train", None) is None:
        raise ValueError("No train/valid split. Call setup_analysis() first.")

    z_train = encode_batched(wf.net, wf.data_train, batch_size=batch_size)
    z_valid = encode_batched(wf.net, wf.data_valid, batch_size=batch_size)
    latent_dim = z_train.shape[1]
    print(
        f"[latent] encoded train {z_train.shape} valid {z_valid.shape} "
        f"(latent_dim={latent_dim})"
    )

    paths = {}
    if save:
        cols = [f"z{i}" for i in range(latent_dim)]
        for name, arr, fname in (
            ("train", z_train, train_csv),
            ("valid", z_valid, valid_csv),
        ):
            path = os.path.join(wf.output_base_dir, fname)
            pd.DataFrame(arr, columns=cols).to_csv(path, index=False)
            paths[name] = path
            print(f"Saved {path}")

    return {
        "z_train": z_train,
        "z_valid": z_valid,
        "latent_dim": latent_dim,
        "csv_train": paths.get("train"),
        "csv_valid": paths.get("valid"),
    }


def reference_vector(z_train, mode="median") -> np.ndarray:
    """Value at which non-scanned latent dimensions are held."""
    z_train = np.asarray(z_train, dtype=np.float64)
    if mode == "median":
        return np.median(z_train, axis=0)
    if mode == "mean":
        return z_train.mean(axis=0)
    if mode == "zeros":
        return np.zeros(z_train.shape[1], dtype=np.float64)
    raise ValueError(f"Unknown reference mode: {mode!r}")


def axis_values(z_all, dim, grid_size=30, padding=0.1) -> np.ndarray:
    """Padded linspace spanning the encoded range of one latent dimension."""
    col = np.asarray(z_all, dtype=np.float64)[:, dim]
    lo, hi = float(col.min()), float(col.max())
    span = hi - lo
    if span == 0.0:
        span = 1.0
    pad = span * padding
    return np.linspace(lo - pad, hi + pad, grid_size)


def scan_pair(
    net,
    z_ref,
    dim_x,
    dim_y,
    xvals,
    yvals,
    *,
    stdval=1.0,
    batch_size=256,
    n_atoms=None,
):
    """RMSD surface over one latent pair, other dimensions fixed at ``z_ref``.

    Returns ``surface`` of shape ``(len(yvals), len(xvals))`` where
    ``surface[iy, ix]`` is the value at ``(xvals[ix], yvals[iy])``.
    """
    z_ref = np.asarray(z_ref, dtype=np.float64)
    xx, yy = np.meshgrid(xvals, yvals)  # both (n_y, n_x)
    grid = np.tile(z_ref, (xx.size, 1))
    grid[:, dim_x] = xx.ravel()
    grid[:, dim_y] = yy.ravel()

    decoded = decode_batched(net, grid, batch_size=batch_size, n_atoms=n_atoms)
    z_again = encode_batched(net, decoded.numpy(), batch_size=batch_size)
    decoded_again = decode_batched(
        net, z_again, batch_size=batch_size, n_atoms=n_atoms
    )

    diff = (decoded - decoded_again).numpy().astype(np.float64)
    rmsd = np.sqrt((diff**2).sum(axis=-1).mean(axis=-1)) * float(stdval)
    return rmsd.reshape(xx.shape)


def scan_all_pairs(
    wf: "AutoencoderWorkflow",
    z_train,
    z_valid=None,
    *,
    dims: Optional[Sequence[int]] = None,
    grid_size: int = 30,
    padding: float = 0.1,
    reference: str = "median",
    batch_size: int = 256,
    npz_filename: str = "latent_landscape_pairs.npz",
    save: bool = True,
):
    """Scan every latent dimension pair as a median slice.

    For ``latent_dim = 6`` this is ``C(6, 2) = 15`` surfaces.
    """
    z_train = np.asarray(z_train, dtype=np.float64)
    z_all = z_train if z_valid is None else np.vstack([z_train, z_valid])
    latent_dim = z_train.shape[1]
    if dims is None:
        dims = list(range(latent_dim))
    dims = list(dims)
    if len(dims) < 2:
        raise ValueError("Need at least two latent dimensions to scan a pair.")

    z_ref = reference_vector(z_train, mode=reference)
    stdval = _resolve_std(wf)
    n_atoms = int(wf.data.dataset.shape[1])
    pairs = list(combinations(dims, 2))
    print(
        f"[latent] scanning {len(pairs)} pair(s) on a {grid_size}x{grid_size} grid; "
        f"other dims held at the {reference} of the training latents"
    )

    surfaces, axes_x, axes_y = {}, {}, {}
    for dim_x, dim_y in pairs:
        xvals = axis_values(z_all, dim_x, grid_size=grid_size, padding=padding)
        yvals = axis_values(z_all, dim_y, grid_size=grid_size, padding=padding)
        surface = scan_pair(
            wf.net,
            z_ref,
            dim_x,
            dim_y,
            xvals,
            yvals,
            stdval=stdval,
            batch_size=batch_size,
            n_atoms=n_atoms,
        )
        key = f"{dim_x}_{dim_y}"
        surfaces[key] = surface
        axes_x[key] = xvals
        axes_y[key] = yvals
        print(
            f"  dims ({dim_x}, {dim_y}): RMSD {surface.min():.3f} - "
            f"{surface.max():.3f} Å"
        )

    scan = {
        "pairs": pairs,
        "surfaces": surfaces,
        "xvals": axes_x,
        "yvals": axes_y,
        "z_ref": z_ref,
        "reference": reference,
        "grid_size": grid_size,
        "latent_dim": latent_dim,
        "npz": None,
    }

    if save:
        npz_path = os.path.join(wf.output_base_dir, npz_filename)
        payload = {"z_ref": z_ref, "pairs": np.asarray(pairs, dtype=int)}
        for key in surfaces:
            payload[f"surface_{key}"] = surfaces[key]
            payload[f"xvals_{key}"] = axes_x[key]
            payload[f"yvals_{key}"] = axes_y[key]
        np.savez_compressed(npz_path, **payload)
        scan["npz"] = npz_path
        print(f"Saved {npz_path}")

    return scan


def load_scan(npz_path):
    """Reload a ``scan_all_pairs`` archive without re-running the model."""
    data = np.load(npz_path)
    pairs = [tuple(int(v) for v in row) for row in data["pairs"]]
    surfaces, axes_x, axes_y = {}, {}, {}
    for dim_x, dim_y in pairs:
        key = f"{dim_x}_{dim_y}"
        surfaces[key] = data[f"surface_{key}"]
        axes_x[key] = data[f"xvals_{key}"]
        axes_y[key] = data[f"yvals_{key}"]
    return {
        "pairs": pairs,
        "surfaces": surfaces,
        "xvals": axes_x,
        "yvals": axes_y,
        "z_ref": data["z_ref"],
        "grid_size": int(next(iter(axes_x.values())).size),
        "npz": npz_path,
    }


def _draw_pair_panel(
    ax,
    scan,
    pair,
    z_train,
    z_valid,
    *,
    vmin,
    vmax,
    cmap,
    point_size,
    train_color,
    valid_color,
    show_legend=False,
):
    dim_x, dim_y = pair
    key = f"{dim_x}_{dim_y}"
    xvals = scan["xvals"][key]
    yvals = scan["yvals"][key]
    surface = scan["surfaces"][key]

    # surface[iy, ix] <-> (xvals[ix], yvals[iy]); pcolormesh consumes exactly
    # this layout, so no transpose and no origin flip.
    mesh = ax.pcolormesh(
        xvals,
        yvals,
        surface,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    ax.scatter(
        z_train[:, dim_x],
        z_train[:, dim_y],
        s=point_size,
        c=train_color,
        marker=".",
        linewidths=0,
        alpha=0.55,
        label="training",
    )
    if z_valid is not None and len(z_valid):
        ax.scatter(
            z_valid[:, dim_x],
            z_valid[:, dim_y],
            s=point_size,
            c=valid_color,
            marker=".",
            linewidths=0,
            alpha=0.85,
            label="validation",
        )
    ax.set_xlabel(f"latent dim {dim_x}")
    ax.set_ylabel(f"latent dim {dim_y}")
    ax.set_xlim(xvals.min(), xvals.max())
    ax.set_ylim(yvals.min(), yvals.max())
    if show_legend:
        ax.legend(fontsize=8, loc="upper right", framealpha=0.85, markerscale=4)
    return mesh


def plot_latent_pair_panels(
    wf: "AutoencoderWorkflow",
    scan,
    z_train,
    z_valid=None,
    *,
    ncols: int = 5,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    point_size: float = 3.0,
    train_color: str = "tab:blue",
    valid_color: str = "tab:red",
    figure_filename: str = "latent_pair_landscapes.png",
    title: Optional[str] = None,
    dpi: int = 200,
    show: bool = True,
):
    """One panel per latent dimension pair: RMSD slice plus train/valid scatter."""
    z_train = np.asarray(z_train, dtype=np.float64)
    z_valid = None if z_valid is None else np.asarray(z_valid, dtype=np.float64)
    pairs = scan["pairs"]

    if vmax is None:
        vmax = float(max(s.max() for s in scan["surfaces"].values()))

    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.3 * nrows), squeeze=False
    )
    axes_flat = list(axes.flatten())

    mesh = None
    for idx, (ax, pair) in enumerate(zip(axes_flat, pairs)):
        mesh = _draw_pair_panel(
            ax,
            scan,
            pair,
            z_train,
            z_valid,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            point_size=point_size,
            train_color=train_color,
            valid_color=valid_color,
            show_legend=(idx == 0),
        )
    for ax in axes_flat[len(pairs) :]:
        ax.set_visible(False)

    if title is None:
        reference = scan.get("reference", "median")
        title = (
            f"Latent-space RMSD landscapes ({scan.get('latent_dim', '?')}D, "
            f"{len(pairs)} projections; other dims at the training {reference})"
        )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 0.93, 0.96])

    cbar_ax = fig.add_axes([0.945, 0.12, 0.012, 0.74])
    fig.colorbar(mesh, cax=cbar_ax, label="decode-encode-decode RMSD [Å]")

    path = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"figure": path, "vmin": vmin, "vmax": vmax}


def plot_latent_pair(
    wf: "AutoencoderWorkflow",
    scan,
    z_train,
    z_valid=None,
    *,
    pair=(0, 1),
    split_panels: bool = True,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    point_size: float = 6.0,
    train_color: str = "tab:blue",
    valid_color: str = "tab:red",
    figure_filename: str = "latent_landscape_pair.png",
    dpi: int = 200,
    show: bool = True,
):
    """Large view of a single latent pair.

    With ``split_panels=True`` the training and validation scatters get their own
    axes over the same surface, matching ``RMSDlandscapesOnePlot`` from the
    single-run notebooks but with the corrected heatmap orientation.
    """
    z_train = np.asarray(z_train, dtype=np.float64)
    z_valid = None if z_valid is None else np.asarray(z_valid, dtype=np.float64)
    pair = tuple(int(v) for v in pair)
    key = f"{pair[0]}_{pair[1]}"
    if key not in scan["surfaces"]:
        raise KeyError(
            f"Pair {pair} was not scanned. Available: {sorted(scan['surfaces'])}"
        )
    if vmax is None:
        vmax = float(scan["surfaces"][key].max())

    if split_panels and z_valid is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        mesh = _draw_pair_panel(
            axes[0], scan, pair, z_train, None,
            vmin=vmin, vmax=vmax, cmap=cmap, point_size=point_size,
            train_color=train_color, valid_color=valid_color,
        )
        axes[0].set_title("Training set")
        _draw_pair_panel(
            axes[1], scan, pair, np.empty((0, z_train.shape[1])), z_valid,
            vmin=vmin, vmax=vmax, cmap=cmap, point_size=point_size,
            train_color=train_color, valid_color=valid_color,
        )
        axes[1].set_title("Validation set")
        fig.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            label="decode-encode-decode RMSD [Å]",
            fraction=0.035,
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        mesh = _draw_pair_panel(
            ax, scan, pair, z_train, z_valid,
            vmin=vmin, vmax=vmax, cmap=cmap, point_size=point_size,
            train_color=train_color, valid_color=valid_color,
            show_legend=True,
        )
        fig.colorbar(mesh, ax=ax, label="decode-encode-decode RMSD [Å]")

    fig.suptitle(
        f"Latent-space RMSD landscape: dims {pair[0]} vs {pair[1]}",
        fontsize=13,
        fontweight="bold",
    )
    path = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"figure": path}


def plot_error_violins(
    wf: "AutoencoderWorkflow",
    *,
    rmsd_csv: Optional[str] = None,
    rmsd_train=None,
    rmsd_valid=None,
    train_color: str = "tab:blue",
    valid_color: str = "tab:red",
    figure_filename: str = "error_violins_train_vs_valid.png",
    dpi: int = 200,
    show: bool = True,
):
    """Train vs validation per-frame reconstruction RMSD as violins.

    Reads the ``rmsd_per_frame_train_valid.csv`` written by
    ``export_kabsch_aligned_datasets`` unless arrays are passed directly.
    """
    if rmsd_train is None or rmsd_valid is None:
        if rmsd_csv is None:
            rmsd_csv = os.path.join(
                wf.output_base_dir, "rmsd_per_frame_train_valid.csv"
            )
        if not os.path.isfile(rmsd_csv):
            raise FileNotFoundError(
                f"{rmsd_csv} not found. Run export_kabsch_aligned_datasets() first."
            )
        df = pd.read_csv(rmsd_csv)
        rmsd_train = df.loc[df["split"] == "train", "rmsd_angstrom"].to_numpy()
        rmsd_valid = df.loc[df["split"] == "valid", "rmsd_angstrom"].to_numpy()

    rmsd_train = np.asarray(rmsd_train, dtype=np.float64)
    rmsd_valid = np.asarray(rmsd_valid, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    parts = ax.violinplot(
        [rmsd_train, rmsd_valid], positions=[1, 2], showmeans=True, showextrema=True
    )
    for body, color in zip(parts["bodies"], (train_color, valid_color)):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.6)
    for part in ("cmeans", "cmins", "cmaxes", "cbars"):
        if part in parts:
            parts[part].set_edgecolor("black")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        [f"training\n(n = {rmsd_train.size})", f"validation\n(n = {rmsd_valid.size})"]
    )
    ax.set_ylabel("Per-frame RMSD [Å]")
    ax.set_title("Reconstruction error: training vs validation")
    ax.grid(True, axis="y", alpha=0.3)

    for pos, vals in ((1, rmsd_train), (2, rmsd_valid)):
        ax.annotate(
            f"mean {vals.mean():.2f}\nmedian {np.median(vals):.2f}",
            xy=(pos, vals.max()),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    path = os.path.join(wf.output_base_dir, figure_filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)

    print(
        f"train  mean={rmsd_train.mean():.4f}  median={np.median(rmsd_train):.4f}  "
        f"max={rmsd_train.max():.4f}"
    )
    print(
        f"valid  mean={rmsd_valid.mean():.4f}  median={np.median(rmsd_valid):.4f}  "
        f"max={rmsd_valid.max():.4f}"
    )
    return {"figure": path, "rmsd_train": rmsd_train, "rmsd_valid": rmsd_valid}
