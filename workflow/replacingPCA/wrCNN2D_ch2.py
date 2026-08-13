"""2-channel writhe + reciprocal-distance-matrix autoencoder ("writheCH2").

This is a variant of :mod:`wrCNN2D` whose encoder ingests a 2-channel
image instead of the single writhe channel:

    channel 0 : writhe matrix  W  (n_seg x n_seg = 73 x 73, zero-padded to
                n_atoms x n_atoms = 74 x 74 so it matches channel 1)
    channel 1 : reciprocal distance matrix  R = 1 / D  (n_atoms x n_atoms,
                diagonal forced to 0)

The decoder is unchanged from :mod:`wrCNN2D`: it still produces coordinates
``X_hat`` of shape ``[B, n_atoms, 3]``. The trainer's composite loss is
``L = writhe_loss + beta * reciprocal_dm_loss`` where the reciprocal-DM term
compares ``rdm(X_hat)`` against ``rdm(X)``.

The encoder operates over ``n_atoms`` (74) with ``base_channels=2`` because
both input channels are padded/built at the atom resolution.
"""

import torch
import torch.nn.functional as F

from .wrCNN2D import AutoEncoder as _WrAutoEncoder
from .wrCNN2D import CoordDecoder, Encoder


class AutoEncoder(_WrAutoEncoder):
    """Writhe + reciprocal-DM (2-channel) encoder, coordinate decoder.

    Reuses the wiggle/torch writhe backends, distance-matrix helper, and
    ``_compute_dims_channels`` from :class:`wrCNN2D.AutoEncoder`; only the
    encoder input (now 2 channels at the atom resolution) and the analysis
    ``encode`` path differ.
    """

    def __init__(self, n_atoms, latent_dim=2, init_c=32, m=2, min_size=9,
                 in_channels=2):
        # NOTE: deliberately skip the parent __init__ (which builds a
        # 1-channel, n_seg-sized encoder) and construct our own modules.
        torch.nn.Module.__init__(self)
        self.n_atoms = int(n_atoms)
        self.n_seg = self.n_atoms - 1
        self.in_channels = int(in_channels)

        # Both encoder channels live at the atom resolution (74), so the
        # encoder is built over n_atoms with base_channels=in_channels.
        enc_dims, enc_channels = self._compute_dims_channels(
            self.n_atoms, init_c, m, min_size, base_channels=self.in_channels
        )
        dec_dims, dec_channels = self._compute_dims_channels(
            self.n_atoms, init_c, m, min_size, base_channels=3
        )
        print(f"writheCH2 encoder: dims={enc_dims}, channels={enc_channels}")
        print(f"writheCH2 decoder: dims={dec_dims}, channels={dec_channels}")
        self.dims = enc_dims
        self.channels = enc_channels
        self.decoder_dims = dec_dims
        self.decoder_channels = dec_channels

        self.encoder = Encoder(latent_dim, enc_dims, enc_channels)
        self.decoder = CoordDecoder(latent_dim, dec_dims, dec_channels)

    @classmethod
    def coords_to_rdm(cls, coord, eps=1e-8):
        """Reciprocal distance matrix ``R = 1 / D`` with zero diagonal.

        ``D`` is the Euclidean distance matrix from
        :meth:`wrCNN2D.AutoEncoder.coords_to_dm` (autograd-safe). The
        reciprocal is taken with a small ``eps`` for numerical safety and
        the (otherwise divergent) diagonal is explicitly masked to 0.

        Args:
            coord: tensor of shape ``[..., n_atoms, 3]``.
            eps: small positive constant added before the reciprocal.

        Returns:
            Tensor of shape ``[..., n_atoms, n_atoms]`` with zero diagonal.
        """
        dm = cls.coords_to_dm(coord)
        n = dm.shape[-1]
        offdiag = 1.0 - torch.eye(n, device=dm.device, dtype=dm.dtype)
        rdm = (1.0 / (dm + eps)) * offdiag
        return rdm

    @staticmethod
    def _pad_writhe(writhe):
        """Zero-pad a writhe image ``[..., n_seg, n_seg]`` to ``n_atoms``.

        Adds one zero row and one zero column at the high-index end so the
        ``73 x 73`` writhe matrix matches the ``74 x 74`` reciprocal DM.
        Accepts inputs with an optional channel dim (``[..., 1, ns, ns]``).
        """
        return F.pad(writhe, (0, 1, 0, 1))

    def _encoder_input(self, coords, writhe):
        """Build the 2-channel encoder input ``[B, 2, n_atoms, n_atoms]``.

        Args:
            coords: ``[B, n_atoms, 3]`` coordinates used for the reciprocal
                distance matrix channel.
            writhe: ``[B, 1, n_seg, n_seg]`` writhe matrix (channel 0).
        """
        writhe_padded = self._pad_writhe(writhe)
        rdm = self.coords_to_rdm(coords).unsqueeze(1)
        return torch.cat([writhe_padded, rdm], dim=1)

    def encode(self, x):
        """Encode coordinates through the 2-channel (writhe + rdm) image.

        Uses the wiggle/numpy writhe backend (no autograd through the
        writhe transform), matching the analysis-path behaviour of
        :class:`wrCNN2D.AutoEncoder.encode` and
        ``MolearnAnalysis.get_encoded``.
        """
        writhe = self.coords_to_writhe(x)
        enc_input = self._encoder_input(x, writhe)
        z = self.encoder(enc_input)
        return z

    def forward(self, x):
        """Full coords -> coords forward pass (autograd-safe).

        Uses the differentiable :func:`coords_to_writhe_torch` and
        :meth:`coords_to_rdm` so gradients can flow from the decoder output
        back through the encoder input when needed.
        """
        writhe = self.coords_to_writhe_torch(x)
        enc_input = self._encoder_input(x, writhe)
        z = self.encoder(enc_input)
        decoded = self.decode(z)
        return decoded
