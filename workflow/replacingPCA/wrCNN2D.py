import math
import os
import sys

import numpy as np
import torch
from torch import nn

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


class Encoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        """
        dm_dim : dimensionality of the input distance matrix
        latent_dim : dimensionality of z
        init_c : number of filters in the first conv block
        m  : channel up-scaling factor
        """

        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"

        self.convs = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=True
                    ),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        final_ch = channels[-1]
        self.finallayer = nn.Linear(final_ch, latent_dim)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        z = self.finallayer(x)
        return z


class CoordDecoder(nn.Module):
    """1D ConvTranspose decoder mapping latent ``z`` to coordinates.

    The decoder mirrors the structure used in
    ``molearn.models.CNN2d_AE.Decoder``: a linear layer expands ``z`` to a
    1D feature map, then a stack of ``ConvTranspose1d`` blocks upsamples it
    until the spatial size matches ``n_atoms``. The final layer outputs 3
    channels (xyz) and the result is permuted to ``[B, n_atoms, 3]``.

    ``dims`` and ``channels`` must be supplied; both are computed from
    ``n_atoms`` (not ``n_seg``) because this decoder operates in
    coordinate space, not writhe space.
    """

    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"

        self.dims = dims
        self.channels = channels

        self.from_latent = nn.Linear(latent_dim, channels[-1] * dims[-1])

        dims_rev = dims[::-1]
        ch_rev = channels[::-1]

        layers = []
        for i in range(len(dims_rev) - 1):
            h_in, h_out = dims_rev[i], dims_rev[i + 1]
            in_ch = ch_rev[i]
            default_out = ch_rev[i + 1]
            is_last = i == len(dims_rev) - 2

            out_ch = 3 if is_last else default_out
            op_h = h_out - 2 * h_in

            layers.append(
                nn.ConvTranspose1d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=op_h,
                    bias=True,
                )
            )
            if not is_last:
                layers.append(nn.BatchNorm1d(out_ch))
                layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), -1)
        h = self.from_latent(z)
        h = h.view(h.size(0), -1, self.dims[-1])
        out = self.convs(h)
        out = out.permute(0, 2, 1).contiguous()
        return out


class AutoEncoder(nn.Module):
    """Writhe-space encoder + coordinate-space decoder.

    Architecture:
        ``X (coords [B, n_atoms, 3]) -> W (writhe [B, 1, n_seg, n_seg])
            -> encoder -> z [B, latent_dim] -> decoder -> X_hat [B, n_atoms, 3]``

    The encoder is a 2D CNN that operates on the (precomputed) writhe
    matrix; the decoder is a 1D transposed CNN that produces back-bone
    coordinates. The recomputed writhe ``W_hat = coords_to_writhe_torch(X_hat)``
    and the input/decoded distance matrices are used in the trainer's
    composite loss.
    """

    def __init__(self, n_atoms, latent_dim=2, init_c=32, m=2, min_size=9):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.n_seg = self.n_atoms - 1

        enc_dims, enc_channels = self._compute_dims_channels(
            self.n_seg, init_c, m, min_size, base_channels=1
        )
        dec_dims, dec_channels = self._compute_dims_channels(
            self.n_atoms, init_c, m, min_size, base_channels=3
        )
        print(f"wrCNN2D encoder: dims={enc_dims}, channels={enc_channels}")
        print(f"wrCNN2D decoder: dims={dec_dims}, channels={dec_channels}")
        self.dims = enc_dims
        self.channels = enc_channels
        self.decoder_dims = dec_dims
        self.decoder_channels = dec_channels

        self.encoder = Encoder(latent_dim, enc_dims, enc_channels)
        self.decoder = CoordDecoder(latent_dim, dec_dims, dec_channels)

    def _compute_dims_channels(self, dim, init_c, m, min_size, base_channels=1):
        dims = [dim]
        channels = [base_channels]
        curr = dim
        ch = init_c

        while curr >= min_size:
            channels.append(ch)
            curr = (curr + 2 * 1 - 4) // 2 + 1
            dims.append(curr)
            ch = int(ch * m)
        return dims, channels

    @staticmethod
    def coords_to_writhe(coord):
        """Standard writhe matrix using the wiggle/numpy backend.

        Fast and bit-identical to prior runs, but **not** connected to the
        autograd graph (uses ``.detach().cpu().numpy()`` internally). Use this
        for inference / analysis paths where no gradient through the writhe
        transform is required.
        """
        coord_np = coord.detach().cpu().numpy().astype(np.float32)
        sigma = find_Sigma_array_batch_parallel(coord_np, num_threads=4)
        sigma_sym = (sigma + np.transpose(sigma, (0, 2, 1))) / 2.0
        writhe = torch.from_numpy(sigma_sym).unsqueeze(1).to(coord.device)
        return writhe

    @staticmethod
    def coords_to_writhe_torch(coord, sharpness=10.0, eps=1e-8):
        """Differentiable, fully-vectorized writhe matrix (PyTorch backend).

        Computes the symmetric matrix Sigma_ij of segment-segment Gauss
        integrals for every pair of non-adjacent backbone segments (i, j) of
        the input chain. All operations are PyTorch tensor ops, so gradients
        flow back to ``coord`` (autograd-safe). Use this whenever you need
        the writhe transform inside a training graph (e.g. computing
        ``Ŵ = coords_to_writhe_torch(x_hat)`` on a decoder output).

        Args:
            coord: tensor of shape ``[..., N_atoms, 3]``. Any number of
                leading batch dimensions is supported.
            sharpness: scale factor inside the tanh-smoothed sign. The
                orientation triple product is first normalized to lie in
                ``[-1, 1]`` (preserving the mathematical scale-invariance of
                the writhe). Higher values are closer to the hard sign;
                lower values give a wider differentiable transition zone.
                On the BRAF activation-loop data the default ``sharpness=10``
                gives ~3% max relative deviation vs the hard sign while
                keeping finite gradients through sign flips; use ~30-100 if
                tighter fidelity is preferred at the cost of a narrower
                gradient window.
            eps: small constant for numerical safety in normalizations.

        Returns:
            Tensor of shape ``[..., 1, N_seg, N_seg]`` (with
            ``N_seg = N_atoms - 1``) that is symmetric, has zero diagonal,
            and zero on adjacent segment pairs.
        """
        n_atoms = coord.shape[-2]
        n_seg = n_atoms - 1
        device = coord.device
        dtype = coord.dtype

        idx = torch.arange(n_seg, device=device)
        i_idx, j_idx = torch.meshgrid(idx, idx, indexing="ij")
        pair_mask = ((i_idx - j_idx).abs() >= 2).to(dtype)

        i_flat = i_idx.flatten()
        j_flat = j_idx.flatten()
        leading = coord.shape[:-2]
        p1 = coord.index_select(-2, i_flat).reshape(*leading, n_seg, n_seg, 3)
        p2 = coord.index_select(-2, i_flat + 1).reshape(*leading, n_seg, n_seg, 3)
        p3 = coord.index_select(-2, j_flat).reshape(*leading, n_seg, n_seg, 3)
        p4 = coord.index_select(-2, j_flat + 1).reshape(*leading, n_seg, n_seg, 3)

        r12 = p2 - p1
        r13 = p3 - p1
        r14 = p4 - p1
        r23 = p3 - p2
        r24 = p4 - p2
        r34 = p4 - p3

        def _safe_normalize(v):
            return v / (v.norm(dim=-1, keepdim=True) + eps)

        n1 = _safe_normalize(torch.linalg.cross(r13, r14, dim=-1))
        n2 = _safe_normalize(torch.linalg.cross(r14, r24, dim=-1))
        n3 = _safe_normalize(torch.linalg.cross(r24, r23, dim=-1))
        n4 = _safe_normalize(torch.linalg.cross(r23, r13, dim=-1))

        asin_clamp = 1.0 - 1e-7
        sigma_star = (
            torch.arcsin((n1 * n2).sum(-1).clamp(-asin_clamp, asin_clamp))
            + torch.arcsin((n2 * n3).sum(-1).clamp(-asin_clamp, asin_clamp))
            + torch.arcsin((n3 * n4).sum(-1).clamp(-asin_clamp, asin_clamp))
            + torch.arcsin((n4 * n1).sum(-1).clamp(-asin_clamp, asin_clamp))
        )

        triple = (torch.linalg.cross(r34, r12, dim=-1) * r13).sum(-1)
        norm_factor = (
            r34.norm(dim=-1) * r12.norm(dim=-1) * r13.norm(dim=-1) + eps
        )
        sign_smooth = torch.tanh(sharpness * triple / norm_factor)

        sigma = sigma_star * sign_smooth / (4.0 * math.pi)
        sigma = sigma * pair_mask
        sigma = 0.5 * (sigma + sigma.transpose(-1, -2))
        return sigma.unsqueeze(-3)

    @staticmethod
    def coords_to_dm(coord, eps=1e-12):
        """Pairwise Euclidean distance matrix from coordinates.

        Uses the matrix identity
        ``D_ij^2 = ||x_i||^2 + ||x_j||^2 - 2 <x_i, x_j>`` followed by a
        positive clamp before the square root, so autograd is well-defined
        even on the (always-zero) diagonal.

        Args:
            coord: tensor of shape ``[..., n_atoms, 3]``.
            eps: small positive constant clamped under the sqrt.

        Returns:
            Tensor of shape ``[..., n_atoms, n_atoms]``.
        """
        gram = torch.matmul(coord, coord.transpose(-1, -2))
        norms_sq = torch.diagonal(gram, dim1=-2, dim2=-1)
        dm_sq = norms_sq.unsqueeze(-1) + norms_sq.unsqueeze(-2) - 2.0 * gram
        return torch.sqrt(torch.clamp(dm_sq, min=eps))

    def encode(self, x):
        """Encode coordinates by going through the writhe matrix.

        ``x`` has shape ``[B, n_atoms, 3]``. The wiggle/numpy backend is
        used here (no autograd through the writhe transform on the encoder
        side); this matches the behaviour of analysis paths such as
        ``MolearnAnalysis.get_encoded`` that just need a forward pass.
        """
        writhe = self.coords_to_writhe(x)
        z = self.encoder(writhe)
        return z

    def decode(self, z):
        """Decode the latent vector ``z`` back to coordinates ``[B, n_atoms, 3]``."""
        return self.decoder(z)

    def forward(self, x):
        """Full coords -> coords forward pass (autograd-safe).

        Unlike :meth:`encode`, this path uses the differentiable
        :func:`coords_to_writhe_torch` so gradients can flow from the
        decoder output back through the writhe transform when the trainer
        supplies a precomputed wiggle target. This is convenient for
        callers that just want ``x_hat = autoencoder(x)``; trainers that
        already hold a precomputed writhe target should call
        ``self.encoder`` and ``self.decode`` directly.
        """
        writhe = self.coords_to_writhe_torch(x)
        z = self.encoder(writhe)
        decoded = self.decode(z)
        return decoded
