"""HA-GCN - Hand-Aware GCN (Song et al., 2024), the sign-language-native model.

Combines a GLOBAL body graph conv with two LOCAL hand sub-graph convs (left/right
21-joint blocks), so handshape topology gets explicit treatment. An adaptive
DropGraph (random joint masking during training) regularises spatial/temporal
redundancy. Multi-stream (joint/bone/motion) is handled by Phase 5 like the rest.
"""
from functools import partial

import torch
import torch.nn as nn

from .base import GCNBackbone
from .ctr_gcn import CTRSpatial
from .st_gcn import STGCSpatial


class HAGCSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A, A_left=None, A_right=None):
        super().__init__()
        self.global_gcn = CTRSpatial(in_channels, out_channels, A)      # whole-body graph
        self.left_gcn = STGCSpatial(in_channels, out_channels, A_left)  # left-hand sub-graph
        self.right_gcn = STGCSpatial(in_channels, out_channels, A_right)  # right-hand sub-graph
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.global_gcn(x) + self.left_gcn(x) + self.right_gcn(x))


class HAModel(nn.Module):
    """Backbone + adaptive DropGraph on the input joints (train-time only)."""

    def __init__(self, backbone, drop_graph=0.1):
        super().__init__()
        self.backbone = backbone
        self.p = drop_graph

    def _dropgraph(self, x):
        N, C, T, V, M = x.shape
        ndrop = max(1, int(self.p * V))
        mask = torch.ones(N, 1, 1, V, 1, device=x.device)
        for n in range(N):
            idx = torch.randperm(V, device=x.device)[:ndrop]
            mask[n, :, :, idx, :] = 0
        return x * mask

    def forward(self, x):
        if self.training and self.p > 0:
            x = self._dropgraph(x)
        return self.backbone(x)


def build_model(num_class, graph, drop_graph=0.1, **kw):
    spatial = partial(HAGCSpatial, A_left=graph.A_left, A_right=graph.A_right)
    backbone = GCNBackbone(spatial, num_class, graph.A, **kw)
    return HAModel(backbone, drop_graph)
