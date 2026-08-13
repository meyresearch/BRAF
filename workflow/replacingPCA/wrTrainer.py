import os
import sys

import matplotlib
import numpy as np
import torch
from molearn.trainers import Trainer
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wiggle"))
try:
    from wiggle.writhe import find_Sigma_array_batch_parallel
except ModuleNotFoundError:
    def _unit_vec(u1, u2):
        cross = np.cross(u1, u2)
        norm = np.linalg.norm(cross)
        if norm == 0:
            return np.zeros_like(cross)
        return cross / norm

    def _gauss_int_4_segment(p1, p2, p3, p4):
        r12 = p2 - p1
        r13 = p3 - p1
        r14 = p4 - p1
        r23 = p3 - p2
        r24 = p4 - p2
        r34 = p4 - p3

        n1 = _unit_vec(r13, r14)
        n2 = _unit_vec(r14, r24)
        n3 = _unit_vec(r24, r23)
        n4 = _unit_vec(r23, r13)

        sigma_star = (
            np.arcsin(np.clip(np.dot(n1, n2), -1.0, 1.0))
            + np.arcsin(np.clip(np.dot(n2, n3), -1.0, 1.0))
            + np.arcsin(np.clip(np.dot(n3, n4), -1.0, 1.0))
            + np.arcsin(np.clip(np.dot(n4, n1), -1.0, 1.0))
        )
        sign = 1.0 if np.dot(np.cross(r34, r12), r13) > 0 else -1.0
        return (1.0 / (4.0 * np.pi)) * sigma_star * sign

    def _find_sigma_array(points):
        segment_num = points.shape[0] - 1
        sigma = np.zeros((segment_num, segment_num), dtype=np.float32)
        for i in range(1, points.shape[0] - 1):
            p1 = points[i]
            p2 = points[i + 1]
            for j in range(0, i - 1):
                p3 = points[j]
                p4 = points[j + 1]
                sigma[i, j] = _gauss_int_4_segment(p1, p2, p3, p4)
        return 2.0 * sigma

    def find_Sigma_array_batch_parallel(trajectory, num_threads=1):
        _ = num_threads
        n_frames = trajectory.shape[0]
        n_seg = trajectory.shape[1] - 1
        out = np.zeros((n_frames, n_seg, n_seg), dtype=np.float32)
        for frame_idx in range(n_frames):
            out[frame_idx] = _find_sigma_array(trajectory[frame_idx])
        return out


class WritheTrainer(Trainer):
    """Trainer for the writhe-space CNN2D autoencoder.

    Forward pipeline (per mini-batch):

        coords X [B, n_atoms, 3]
            -> wiggle writhe W [B, 1, n_seg, n_seg]   (precomputed in set_data)
            -> encoder -> z [B, latent_dim]
            -> decoder -> X_hat [B, n_atoms, 3]
            -> coords_to_writhe_torch -> W_hat [B, 1, n_seg, n_seg]   (autograd)
            -> coords_to_dm -> D, D_hat                                (autograd)

    Composite loss:

        L = MSE_offdiag(W_hat - W) + beta * MSE(D_hat - D)

    The first term enforces topological agreement of the reconstruction
    (the writhe matrix is gauge-invariant: it is unchanged under
    translations, rotations, uniform scalings and reflections of the
    chain). The second term breaks scale and reflection invariance by
    constraining the pairwise Euclidean geometry of the decoder output to
    match the input. ``beta`` interpolates between the gauge-invariant
    objective (``beta=0``, equivalent to the previous wrCNN2D loss) and
    a coordinate-anchored one.
    """

    def __init__(self, device=None, beta=0.0, **kwargs):
        super().__init__(device=device, **kwargs)
        self.beta = float(beta)

    def set_data(self, data, **kwargs):
        super().set_data(data, **kwargs)

        coords_std = self._data.dataset.detach().cpu().float()
        coords_real = coords_std * float(self.std) + float(self.mean)
        coords_np = coords_real.numpy().astype(np.float32, copy=False)

        sigma = find_Sigma_array_batch_parallel(coords_np, num_threads=4)
        sigma_sym = (sigma + np.transpose(sigma, (0, 2, 1))) / 2.0
        writhe_all = torch.from_numpy(sigma_sym).unsqueeze(1).float()

        train_idx = self._data.train_indices
        valid_idx = self._data.valid_indices
        coords_train = coords_std[train_idx]
        coords_valid = coords_std[valid_idx]
        writhe_train = writhe_all[train_idx]
        writhe_valid = writhe_all[valid_idx]

        self.coords_train = coords_train
        self.coords_valid = coords_valid
        self.writhe_train = writhe_train
        self.writhe_valid = writhe_valid
        self.n_seg = int(sigma_sym.shape[-1])

        batch_size = kwargs.get("batch_size", 8)
        self.train_dataloader = DataLoader(
            TensorDataset(coords_train, writhe_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.valid_dataloader = DataLoader(
            TensorDataset(coords_valid, writhe_valid),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        print(
            f"Pre-computed writhe matrices: {writhe_all.shape}, "
            f"train={writhe_train.shape}, valid={writhe_valid.shape}, "
            f"beta={self.beta:g}"
        )

    def _run_phase(
        self,
        dataloader,
        step_fn,
        prefix,
        *,
        backward,
        dry_run=False,
    ):
        """Override that pulls *both* coords and writhe out of each batch.

        The base ``Trainer._run_phase`` only takes ``batch[0]``. Our
        dataloader yields a ``(coords, writhe)`` pair per item, so we
        forward them together to ``step_fn`` as a tuple.
        """
        totals = {}
        count = 0
        for batch_pair in dataloader:
            coords_batch = batch_pair[0].to(self.device)
            writhe_batch = batch_pair[1].to(self.device)
            batch_size = coords_batch.shape[0]
            if backward and not dry_run:
                self.optimiser.zero_grad()
                outputs = step_fn((coords_batch, writhe_batch))
                outputs["loss"].backward()
                self.optimiser.step()
            else:
                with torch.no_grad():
                    outputs = step_fn((coords_batch, writhe_batch))

            for key, value in outputs.items():
                totals[key] = totals.get(key, 0.0) + value.item() * batch_size
            count += batch_size

        if count == 0:
            return {}, {}

        averaged = {key: totals[key] / count for key in totals}
        prefixed = {f"{prefix}_{key}": averaged[key] for key in averaged}
        return prefixed, averaged

    def common_step(self, batch):
        """Forward pass and composite loss for one mini-batch.

        ``batch`` is a ``(coords, writhe)`` tuple yielded by
        :meth:`_run_phase`. ``coords`` are standardized
        ``(x - mean)/std`` because both the wiggle writhe (computed in
        real space inside :meth:`set_data`) and the recomputed
        :func:`coords_to_writhe_torch` agree on standardized inputs --
        the writhe matrix is invariant under the global affine
        ``x -> std*x + mean``.
        """
        self._internal = {}
        coords_batch, writhe_batch = batch

        z = self.autoencoder.encoder(writhe_batch)
        self._internal["encoded"] = z
        decoded_coords = self.autoencoder.decode(z)
        self._internal["decoded"] = decoded_coords

        # W_hat from decoded coords (autograd-safe).
        decoded_writhe = self.autoencoder.coords_to_writhe_torch(decoded_coords)

        n = writhe_batch.shape[-1]
        offdiag = (
            (1.0 - torch.eye(n, device=writhe_batch.device, dtype=writhe_batch.dtype))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sq_diff_w = (decoded_writhe - writhe_batch) ** 2 * offdiag
        n_active = offdiag.sum()
        writhe_loss = sq_diff_w.sum() / (n_active * writhe_batch.shape[0])

        # D, D_hat in standardized coordinate space.
        target_dm = self.autoencoder.coords_to_dm(coords_batch)
        decoded_dm = self.autoencoder.coords_to_dm(decoded_coords)
        dm_loss = ((decoded_dm - target_dm) ** 2).mean()

        total_loss = writhe_loss + self.beta * dm_loss
        return dict(writhe_loss=writhe_loss, dm_loss=dm_loss, loss=total_loss)

    def train_step(self, batch):
        return self.common_step(batch)

    def valid_step(self, batch):
        return self.common_step(batch)

    def _batched_encode_decode(self, writhe, coords=None, n_frames=None, batch_size=8):
        """Encode/decode ``n_frames`` writhe inputs in GPU-safe mini-batches."""
        n_total = len(writhe)
        n_use = n_total if n_frames is None else min(int(n_frames), n_total)
        writhe = writhe[:n_use]
        if coords is not None:
            coords = coords[:n_use]

        decoded_writhe_chunks = []
        decoded_coords_chunks = [] if coords is not None else None
        target_dm_chunks = [] if coords is not None else None

        self.autoencoder.eval()
        with torch.no_grad():
            for start in range(0, n_use, batch_size):
                w_batch = writhe[start : start + batch_size].to(self.device)
                z = self.autoencoder.encoder(w_batch)
                decoded_coords = self.autoencoder.decode(z)
                decoded_writhe_chunks.append(
                    self.autoencoder.coords_to_writhe_torch(decoded_coords).cpu()
                )
                if coords is not None:
                    c_batch = coords[start : start + batch_size].to(self.device)
                    target_dm_chunks.append(self.autoencoder.coords_to_dm(c_batch).cpu())
                    decoded_coords_chunks.append(
                        self.autoencoder.coords_to_dm(decoded_coords).cpu()
                    )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        decoded_writhe = torch.cat(decoded_writhe_chunks, dim=0)
        if coords is None:
            return decoded_writhe, None, None
        return decoded_writhe, torch.cat(target_dm_chunks, dim=0), torch.cat(
            decoded_coords_chunks, dim=0
        )

    def generate_reconstruction_gif(
        self, output_path, dataset="train", fps=3, max_frames=None, batch_size=8
    ):
        """Visualize input writhe vs writhe recomputed from decoded coords.

        With the new architecture the decoder produces coordinates, so the
        "reconstructed writhe" panel here is
        ``coords_to_writhe_torch(autoencoder.decode(z))`` -- the same
        quantity the writhe-loss term compares against.
        """
        if dataset not in {"train", "valid"}:
            raise ValueError("dataset must be 'train' or 'valid'")

        coords = self.coords_train if dataset == "train" else self.coords_valid
        writhe = self.writhe_train if dataset == "train" else self.writhe_valid
        if writhe is None or len(writhe) == 0:
            raise ValueError(f"No data available for dataset='{dataset}'")

        n_total = len(writhe)
        n_plot = n_total if max_frames is None else min(int(max_frames), n_total)
        decoded_writhe = self._batched_encode_decode(
            writhe, n_frames=n_plot, batch_size=batch_size
        )[0]
        target = writhe[:n_plot].cpu()

        combined = torch.cat([target[:n_plot], decoded_writhe[:n_plot]], dim=0)
        vmin = float(combined.min().item())
        vmax = float(combined.max().item())

        n = target.shape[-1]
        mask = (1.0 - torch.eye(n)).unsqueeze(0).unsqueeze(0)
        frames = []

        for i in range(n_plot):
            inp = target[i, 0]
            rec = decoded_writhe[i, 0]
            diff = torch.abs(rec - inp)
            frame_mse = (((rec - inp) ** 2) * mask[0, 0]).sum() / mask[0, 0].sum()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axes[0].imshow(inp.numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            im1 = axes[1].imshow(rec.numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            im2 = axes[2].imshow(diff.numpy(), cmap="viridis")

            axes[0].set_title("Input writhe (wiggle)")
            axes[1].set_title("Recomputed from decoded coords")
            axes[2].set_title("Absolute diff")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(f"{dataset} frame {i} | masked MSE={float(frame_mse):.6f}")
            fig.colorbar(im0, ax=[axes[0], axes[1]], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            fig.tight_layout()

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[:, :, :3].copy())
            plt.close(fig)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if imageio is not None:
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
            print(f"Saved reconstruction GIF: {output_path} ({len(frames)} frames)")
        else:
            png_dir = os.path.splitext(output_path)[0] + "_frames"
            os.makedirs(png_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(os.path.join(png_dir, f"frame_{i:04d}.png"), frame)
            print(
                "imageio is unavailable; saved PNG frames instead to "
                f"{png_dir} ({len(frames)} frames)"
            )

    def generate_dm_reconstruction_gif(
        self, output_path, dataset="train", fps=3, max_frames=None, batch_size=8
    ):
        """Visualize input distance matrix vs DM recomputed from decoded coords.

        Parallel to :py:meth:`generate_reconstruction_gif`, but for the
        distance-matrix loss term: the right panel is
        ``coords_to_dm(autoencoder.decode(z))`` -- the same quantity the
        ``dm_loss`` term compares against ``coords_to_dm(input coords)``.
        DMs are shown in *standardized* coordinate space (the space the
        loss is minimized in), so the per-frame MSE printed in each title
        is directly comparable to the ``valid_dm_loss`` curve in the
        training history.
        """
        if dataset not in {"train", "valid"}:
            raise ValueError("dataset must be 'train' or 'valid'")

        coords = self.coords_train if dataset == "train" else self.coords_valid
        writhe = self.writhe_train if dataset == "train" else self.writhe_valid
        if coords is None or len(coords) == 0:
            raise ValueError(f"No data available for dataset='{dataset}'")

        n_total = len(coords)
        n_plot = n_total if max_frames is None else min(int(max_frames), n_total)
        _, target_dm, decoded_dm = self._batched_encode_decode(
            writhe, coords=coords, n_frames=n_plot, batch_size=batch_size
        )

        combined = torch.cat([target_dm[:n_plot], decoded_dm[:n_plot]], dim=0)
        vmin = float(combined.min().item())
        vmax = float(combined.max().item())

        frames = []
        for i in range(n_plot):
            inp = target_dm[i]
            rec = decoded_dm[i]
            diff = torch.abs(rec - inp)
            frame_mse = ((rec - inp) ** 2).mean()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axes[0].imshow(inp.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            im1 = axes[1].imshow(rec.numpy(), cmap="viridis", vmin=vmin, vmax=vmax)
            im2 = axes[2].imshow(diff.numpy(), cmap="magma")

            axes[0].set_title("Input distance matrix")
            axes[1].set_title("Recomputed from decoded coords")
            axes[2].set_title("Absolute diff")
            for ax in axes:
                ax.set_xticks([])
                ax.set_yticks([])

            fig.suptitle(
                f"{dataset} frame {i} | DM MSE = {float(frame_mse):.6f} "
                "(standardized coord units)"
            )
            fig.colorbar(im0, ax=[axes[0], axes[1]], fraction=0.046, pad=0.04)
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            fig.tight_layout()

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frames.append(rgba[:, :, :3].copy())
            plt.close(fig)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if imageio is not None:
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
            print(f"Saved DM reconstruction GIF: {output_path} ({len(frames)} frames)")
        else:
            png_dir = os.path.splitext(output_path)[0] + "_frames"
            os.makedirs(png_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.imsave(os.path.join(png_dir, f"frame_{i:04d}.png"), frame)
            print(
                "imageio is unavailable; saved PNG frames instead to "
                f"{png_dir} ({len(frames)} frames)"
            )


class WritheCH2Trainer(WritheTrainer):
    """Trainer for the 2-channel writhe + reciprocal-DM autoencoder.

    Identical data pipeline to :class:`WritheTrainer` (``set_data``
    precomputes the wiggle writhe and the dataloader yields
    ``(coords, writhe)`` pairs), but the encoder now consumes a 2-channel
    image built on the fly from the (zero-padded) writhe matrix and the
    reciprocal distance matrix of the input coordinates.

    Composite loss::

        L = MSE_offdiag(W_hat - W) + beta * MSE(R_hat - R)

    where ``R = 1 / D`` is the reciprocal distance matrix (zero diagonal)
    and ``W`` is the gauge-invariant writhe matrix. The reciprocal-DM term
    replaces the plain distance-matrix term used in :class:`WritheTrainer`;
    it is still reported under the ``dm_loss`` key so the existing
    log/plotting code (``plot_training_history``) works unchanged.
    """

    def common_step(self, batch):
        """Forward pass and composite loss for one mini-batch.

        ``batch`` is a ``(coords, writhe)`` tuple yielded by
        :meth:`WritheTrainer._run_phase`. The encoder input is the
        2-channel ``[B, 2, n_atoms, n_atoms]`` image assembled by
        :meth:`wrCNN2D_ch2.AutoEncoder._encoder_input`.
        """
        self._internal = {}
        coords_batch, writhe_batch = batch

        enc_input = self.autoencoder._encoder_input(coords_batch, writhe_batch)
        z = self.autoencoder.encoder(enc_input)
        self._internal["encoded"] = z
        decoded_coords = self.autoencoder.decode(z)
        self._internal["decoded"] = decoded_coords

        # W_hat from decoded coords (autograd-safe), compared on the
        # unpadded n_seg x n_seg writhe grid.
        decoded_writhe = self.autoencoder.coords_to_writhe_torch(decoded_coords)

        n = writhe_batch.shape[-1]
        offdiag = (
            (1.0 - torch.eye(n, device=writhe_batch.device, dtype=writhe_batch.dtype))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sq_diff_w = (decoded_writhe - writhe_batch) ** 2 * offdiag
        n_active = offdiag.sum()
        writhe_loss = sq_diff_w.sum() / (n_active * writhe_batch.shape[0])

        # Reciprocal distance matrices in standardized coordinate space.
        target_rdm = self.autoencoder.coords_to_rdm(coords_batch)
        decoded_rdm = self.autoencoder.coords_to_rdm(decoded_coords)
        dm_loss = ((decoded_rdm - target_rdm) ** 2).mean()

        total_loss = writhe_loss + self.beta * dm_loss
        return dict(writhe_loss=writhe_loss, dm_loss=dm_loss, loss=total_loss)


def build_local_weight_matrix(n, device, local_k=4, local_weight=0.9, nonlocal_weight=0.1):
    """Build a [1, 1, n, n] weight matrix (molearn AE_DM_Trainer convention)."""
    idx = torch.arange(n, device=device)
    offs = torch.abs(idx[:, None] - idx[None, :])
    local_mask = offs <= int(local_k)
    w = torch.full((n, n), fill_value=float(nonlocal_weight), device=device)
    w[local_mask] = float(local_weight)
    return w.unsqueeze(0).unsqueeze(0)


def weighted_matrix_loss(pred, target, weight_matrix):
    """
    Local-weighted matrix loss matching molearn AE_DM_Trainer._get_dm_loss.

    ``pred`` and ``target`` may be ``[B, n, n]`` or ``[B, 1, n, n]``.
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    diff = pred - target
    wsq = diff.pow(2) * weight_matrix
    per_sample = wsq.sum(dim=(1, 2, 3)).sqrt()
    return per_sample.mean()


class LocalWeightedDMMixin:
    """Shared local_k weighting for distance / reciprocal-DM loss terms."""

    local_k: int = 4
    local_weight: float = 0.9
    nonlocal_weight: float = 0.1

    def _init_local_dm_weights(self, local_k=4, local_weight=0.9, nonlocal_weight=0.1):
        self.local_k = int(local_k)
        self.local_weight = float(local_weight)
        self.nonlocal_weight = float(nonlocal_weight)
        self.W = None

    def _build_local_weight_matrix(self, n_atoms):
        self.W = build_local_weight_matrix(
            n_atoms,
            self.device,
            local_k=self.local_k,
            local_weight=self.local_weight,
            nonlocal_weight=self.nonlocal_weight,
        )

    def _weighted_matrix_loss(self, pred, target):
        if self.W is None:
            raise RuntimeError("Local weight matrix not built; call set_data first.")
        return weighted_matrix_loss(pred, target, self.W)


class CNN2d_LocalDM_Trainer(LocalWeightedDMMixin, Trainer):
    """2D CNN AE trainer with local_k-weighted distance-matrix loss (2DAEv2).

    Same architecture as the standard CNN2d autoencoder (DM in, coords out),
    but the training objective matches molearn's AE_DM_Trainer weighted DM
    term instead of flat coordinate MSE.
    """

    def __init__(
        self,
        device=None,
        local_k=4,
        local_weight=0.9,
        nonlocal_weight=0.1,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)
        self._init_local_dm_weights(local_k, local_weight, nonlocal_weight)

    def set_data(self, data, **kwargs):
        super().set_data(data, **kwargs)
        n_atoms = int(self._data.dataset.shape[1])
        self._build_local_weight_matrix(n_atoms)
        print(
            f"Local DM weighting: n_atoms={n_atoms}, local_k={self.local_k}, "
            f"local_weight={self.local_weight:g}, nonlocal_weight={self.nonlocal_weight:g}"
        )

    def common_step(self, batch):
        self._internal = {}
        dm_batch = self.autoencoder.coords_to_dm(batch)
        z = self.autoencoder.encoder(dm_batch)
        self._internal["encoded"] = z
        decoded = self.autoencoder.decode(z)[:, : batch.size(1), :]
        self._internal["decoded"] = decoded
        dm_decoded = self.autoencoder.coords_to_dm(decoded)
        dm_loss = self._weighted_matrix_loss(dm_decoded, dm_batch)
        return dict(dm_loss=dm_loss, loss=dm_loss)

    def train_step(self, batch):
        return self.common_step(batch)

    def valid_step(self, batch):
        return self.common_step(batch)


class WritheCH2LocalTrainer(LocalWeightedDMMixin, WritheCH2Trainer):
    """writheCH2 trainer with local_k-weighted reciprocal-DM loss (wrAEv3).

    Identical to :class:`WritheCH2Trainer` except the geometric term uses
    molearn-style local upweighting on the reciprocal distance matrix.
    """

    def __init__(
        self,
        device=None,
        beta=1.0,
        local_k=4,
        local_weight=0.9,
        nonlocal_weight=0.1,
        **kwargs,
    ):
        super().__init__(device=device, beta=beta, **kwargs)
        self._init_local_dm_weights(local_k, local_weight, nonlocal_weight)

    def set_data(self, data, **kwargs):
        super().set_data(data, **kwargs)
        n_atoms = int(self.coords_train.shape[1])
        self._build_local_weight_matrix(n_atoms)
        print(
            f"Local RDM weighting: n_atoms={n_atoms}, local_k={self.local_k}, "
            f"local_weight={self.local_weight:g}, nonlocal_weight={self.nonlocal_weight:g}, "
            f"beta={self.beta:g}"
        )

    def common_step(self, batch):
        self._internal = {}
        coords_batch, writhe_batch = batch

        enc_input = self.autoencoder._encoder_input(coords_batch, writhe_batch)
        z = self.autoencoder.encoder(enc_input)
        self._internal["encoded"] = z
        decoded_coords = self.autoencoder.decode(z)
        self._internal["decoded"] = decoded_coords

        decoded_writhe = self.autoencoder.coords_to_writhe_torch(decoded_coords)

        n = writhe_batch.shape[-1]
        offdiag = (
            (1.0 - torch.eye(n, device=writhe_batch.device, dtype=writhe_batch.dtype))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        sq_diff_w = (decoded_writhe - writhe_batch) ** 2 * offdiag
        n_active = offdiag.sum()
        writhe_loss = sq_diff_w.sum() / (n_active * writhe_batch.shape[0])

        target_rdm = self.autoencoder.coords_to_rdm(coords_batch)
        decoded_rdm = self.autoencoder.coords_to_rdm(decoded_coords)
        dm_loss = self._weighted_matrix_loss(decoded_rdm, target_rdm)

        total_loss = writhe_loss + self.beta * dm_loss
        return dict(writhe_loss=writhe_loss, dm_loss=dm_loss, loss=total_loss)
