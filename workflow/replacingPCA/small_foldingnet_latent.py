"""Small foldingnet AE with configurable latent dimension.

Upstream ``molearn.models.small_foldingnet.Small_AutoEncoder`` hardcodes
``x.view(-1, 2, 1)`` in ``Small_Decoder``, so latent-dimension scans crash
for ``d != 2``. This module keeps the same Small encoder/decoder stack but
uses ``latent_dimension`` throughout. At ``latent_dimension=2`` it matches
the stock December-workflow Small model.
"""

from __future__ import annotations

from torch import nn

from molearn.models.foldingnet import AutoEncoder, Decoder_Layer, Encoder


class Small_Decoder(nn.Module):
    """Small FoldingNet decoder with configurable latent width."""

    def __init__(self, out_points, latent_dimension=2, **kwargs):
        del kwargs  # accept unused molearn/Trainer kwargs
        super().__init__()
        self.out_points = out_points
        self.latent_dimension = int(latent_dimension)
        start_out = (out_points // 8) + 1
        self.layer1 = Decoder_Layer(
            1, start_out, self.latent_dimension, 3 * 32
        )
        self.layer2 = Decoder_Layer(start_out, start_out * 8, 3 * 32, 3)

    def forward(self, x):
        x = x.view(-1, self.latent_dimension, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class Small_AutoEncoder(AutoEncoder):
    """Small FoldingNet AE (graph encoder + small decoder) with latent_dimension."""

    def __init__(self, out_points, latent_dimension=2, **kwargs):
        # Skip foldingnet.AutoEncoder.__init__ (full Decoder); init nn.Module only.
        super(AutoEncoder, self).__init__()
        latent_dimension = int(latent_dimension)
        self.encoder = Encoder(latent_dimension=latent_dimension, **kwargs)
        self.decoder = Small_Decoder(
            out_points, latent_dimension=latent_dimension, **kwargs
        )
