"""HD-GCN (Lee et al., ICCV 2023) spatial module.

Hierarchically Decomposed Graph Convolution: the graph is split into hierarchical
edge sets by hop distance, each convolved separately, then aggregated with a
learned attention weight per hierarchy (an attention-guided decomposition).
"""
import numpy as np
import torch
import torch.nn as nn

from .base import GCNBackbone, conv_init, bn_init
from ._adj import to_numpy_binary, k_adjacency, normalize_adj


class HDGCSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A, num_hier=3):
        super().__init__()
        A_bin = to_numpy_binary(A)
        mats = [np.eye(A_bin.shape[0], dtype=np.float32)]              # self
        mats += [normalize_adj(k_adjacency(A_bin, k)) for k in range(1, num_hier + 1)]
        self.register_buffer("As", torch.tensor(np.stack(mats), dtype=torch.float32))
        self.num = len(mats)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 1) for _ in range(self.num)])
        self.att = nn.Parameter(torch.ones(self.num))                 # hierarchy attention
        self.bn = nn.BatchNorm2d(out_channels)
        for m in self.convs:
            conv_init(m)
        bn_init(self.bn)

    def forward(self, x):
        w = torch.softmax(self.att, 0)
        out = 0
        for i in range(self.num):
            y = self.convs[i](x)
            y = torch.einsum("nctv,vw->nctw", y, self.As[i])
            out = out + w[i] * y
        return self.bn(out)


def build_model(num_class, graph, **kw):
    return GCNBackbone(HDGCSpatial, num_class, graph.A, **kw)
