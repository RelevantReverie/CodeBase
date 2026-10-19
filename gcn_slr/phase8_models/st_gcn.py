"""ST-GCN (Yan et al., AAAI 2018) spatial module.

Fixed partitioned graph (self / inward / outward) with a learnable edge-importance
mask. For a fair comparison it uses the shared multi-scale temporal conv from the
backbone (the original paper used a single-scale TCN).
"""
import numpy as np
import torch
import torch.nn as nn

from .base import GCNBackbone, conv_init, bn_init


class STGCSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        A = torch.as_tensor(np.asarray(A if not torch.is_tensor(A) else A.cpu()),
                            dtype=torch.float32)
        self.K = A.shape[0]
        self.out_channels = out_channels
        self.register_buffer("A", A)
        self.edge = nn.Parameter(torch.ones_like(A))      # learnable edge importance
        self.conv = nn.Conv2d(in_channels, out_channels * self.K, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn)

    def forward(self, x):
        N, C, T, V = x.shape
        x = self.conv(x).view(N, self.K, self.out_channels, T, V)
        A = self.A * self.edge
        x = torch.einsum("nkctv,kvw->nctw", x, A)
        return self.bn(x)


def build_model(num_class, graph, **kw):
    return GCNBackbone(STGCSpatial, num_class, graph.A, **kw)
