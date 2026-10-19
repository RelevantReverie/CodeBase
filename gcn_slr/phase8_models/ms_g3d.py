"""MS-G3D (Liu et al., CVPR 2020) spatial module.

Core idea: a *disentangled multi-scale* graph conv - separate exactly-k-hop
adjacency matrices so distant joints aren't over-weighted. (The full G3D
spatial-temporal sliding-window factor is omitted for a uniform temporal path;
the multi-scale spatial aggregation, the paper's main contribution, is kept.)
"""
import numpy as np
import torch
import torch.nn as nn

from .base import GCNBackbone, conv_init, bn_init
from ._adj import to_numpy_binary, k_adjacency, normalize_adj


class MSGCSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_scales=4):
        super().__init__()
        A_bin = to_numpy_binary(A)
        mats = [normalize_adj(k_adjacency(A_bin, k)) for k in range(num_scales)]
        self.register_buffer("As", torch.tensor(np.stack(mats), dtype=torch.float32))
        self.num_scales = num_scales
        self.conv = nn.Conv2d(in_channels * num_scales, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn)

    def forward(self, x):
        feats = [torch.einsum("nctv,vw->nctw", x, self.As[s]) for s in range(self.num_scales)]
        x = torch.cat(feats, dim=1)
        return self.bn(self.conv(x))


def build_model(num_class, graph, **kw):
    return GCNBackbone(MSGCSpatial, num_class, graph.A, **kw)
