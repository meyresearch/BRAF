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
    def set_data(self, data, **kwargs):
        super().set_data(data, **kwargs)

        coords_std = self._data.dataset.detach().cpu()
        coords_real = coords_std * float(self.std) + float(self.mean)
        coords_np = coords_real.numpy().astype(np.float32, copy=False)

        sigma = find_Sigma_array_batch_parallel(coords_np, num_threads=4)
        sigma_sym = (sigma + np.transpose(sigma, (0, 2, 1))) / 2.0
        writhe_all = torch.from_numpy(sigma_sym).unsqueeze(1).float()

        train_idx = self._data.train_indices
        valid_idx = self._data.valid_indices
        writhe_train = writhe_all[train_idx]
        writhe_valid = writhe_all[valid_idx]

        self.writhe_train = writhe_train
        self.writhe_valid = writhe_valid
        self.n_seg = int(sigma_sym.shape[-1])

        self._coord_train_dataloader = self.train_dataloader
        self._coord_valid_dataloader = self.valid_dataloader

        batch_size = kwargs.get("batch_size", 8)
        self.train_dataloader = DataLoader(
            TensorDataset(writhe_train),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        self.valid_dataloader = DataLoader(
            TensorDataset(writhe_valid),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )

        print(
            f"Pre-computed writhe matrices: {writhe_all.shape}, "
            f"train={writhe_train.shape}, valid={writhe_valid.shape}"
        )

    def common_step(self, batch):
        self._internal = {}
        encoded = self.autoencoder.encoder(batch)
        self._internal["encoded"] = encoded
        decoded = self.autoencoder.decode(encoded)
        self._internal["decoded"] = decoded

        n = batch.shape[-1]
        mask = (1.0 - torch.eye(n, device=batch.device)).unsqueeze(0).unsqueeze(0)
        sq_diff = (decoded - batch) ** 2 * mask
        n_active = mask.sum()
        loss = sq_diff.sum() / (n_active * batch.shape[0])
        return dict(writhe_loss=loss)

    def train_step(self, batch):
        results = self.common_step(batch)
        results["loss"] = results["writhe_loss"]
        return results

    def valid_step(self, batch):
        results = self.common_step(batch)
        results["loss"] = results["writhe_loss"]
        return results

    def generate_reconstruction_gif(
        self, output_path, dataset="train", fps=3, max_frames=None
    ):
        if dataset not in {"train", "valid"}:
            raise ValueError("dataset must be 'train' or 'valid'")

        data = self.writhe_train if dataset == "train" else self.writhe_valid
        if data is None or len(data) == 0:
            raise ValueError(f"No data available for dataset='{dataset}'")

        self.autoencoder.eval()
        with torch.no_grad():
            batch = data.to(self.device)
            encoded = self.autoencoder.encoder(batch)
            decoded = self.autoencoder.decode(encoded).cpu()

        target = data.cpu()
        n_total = target.shape[0]
        n_plot = n_total if max_frames is None else min(int(max_frames), n_total)

        combined = torch.cat([target[:n_plot], decoded[:n_plot]], dim=0)
        vmin = float(combined.min().item())
        vmax = float(combined.max().item())

        n = target.shape[-1]
        mask = (1.0 - torch.eye(n)).unsqueeze(0).unsqueeze(0)
        frames = []

        for i in range(n_plot):
            inp = target[i, 0]
            rec = decoded[i, 0]
            diff = torch.abs(rec - inp)
            frame_mse = (((rec - inp) ** 2) * mask[0, 0]).sum() / mask[0, 0].sum()

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            im0 = axes[0].imshow(inp.numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            im1 = axes[1].imshow(rec.numpy(), cmap="RdBu_r", vmin=vmin, vmax=vmax)
            im2 = axes[2].imshow(diff.numpy(), cmap="viridis")

            axes[0].set_title("Input writhe")
            axes[1].set_title("Reconstructed writhe")
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
