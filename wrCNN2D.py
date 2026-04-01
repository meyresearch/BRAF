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


class Decoder2D(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        assert len(dims) == len(channels), "dims/channels length mismatch"

        self.from_latent = nn.Linear(latent_dim, channels[-1] * dims[-1] * dims[-1])
        self.dims = dims

        dims_rev = dims[::-1]
        ch_rev = channels[::-1]

        layers = []
        for i in range(len(dims_rev) - 1):
            h_in, h_out = dims_rev[i], dims_rev[i + 1]
            in_ch = ch_rev[i]
            default_out = ch_rev[i + 1]
            is_last = i == len(dims_rev) - 2

            out_ch = 1 if is_last else default_out
            op_h = h_out - 2 * h_in

            layers.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    output_padding=(op_h, op_h),
                    bias=True,
                )
            )
            if not is_last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.LeakyReLU(0.1, inplace=True))

        self.convs = nn.Sequential(*layers)

    def forward(self, z):
        z = z.view(z.size(0), -1)
        h = self.from_latent(z)
        h = h.view(h.size(0), -1, self.dims[-1], self.dims[-1])
        out = self.convs(h)
        out = 0.5 * (out + out.transpose(-2, -1))
        return out


class AutoEncoder(nn.Module):
    def __init__(self, n_atoms, latent_dim=2, init_c=32, m=2, min_size=9):
        super().__init__()
        self.n_atoms = int(n_atoms)
        self.n_seg = self.n_atoms - 1

        dims, channels = self._compute_dims_channels(self.n_seg, init_c, m, min_size)
        print(f"wrCNN2D: dims={dims}, channels={channels}")
        self.dims = dims
        self.channels = channels

        self.encoder = Encoder(latent_dim, dims, channels)
        self.decoder = Decoder2D(latent_dim, dims, channels)

    def _compute_dims_channels(self, dim, init_c, m, min_size):
        dims = [dim]
        channels = [1]
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
        coord_np = coord.detach().cpu().numpy().astype(np.float32)
        sigma = find_Sigma_array_batch_parallel(coord_np, num_threads=4)
        sigma_sym = (sigma + np.transpose(sigma, (0, 2, 1))) / 2.0
        writhe = torch.from_numpy(sigma_sym).unsqueeze(1).to(coord.device)
        return writhe

    def encode(self, x):
        writhe = self.coords_to_writhe(x)
        z = self.encoder(writhe)
        return z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        decoded = self.decode(z)
        return decoded
