"""Shared building blocks + a common 10-block GCN backbone.

Every model (Phase 8) plugs its own *spatial* module into this backbone, so the
six nets differ only in how they aggregate over the skeleton graph - everything
external (input tensor, temporal conv, depth, pooling, head) is identical, which
is what makes the comparison fair.

A spatial module has signature:  SpatialModule(in_channels, out_channels, A)
and forward(x) with x: [N, C, T, V] -> [N, C_out, T, V].
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# init helpers
# --------------------------------------------------------------------------- #
def conv_init(m):
    if getattr(m, "weight", None) is not None:
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
    if getattr(m, "bias", None) is not None:
        nn.init.constant_(m.bias, 0)


def bn_init(bn, scale=1.0):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


# --------------------------------------------------------------------------- #
# temporal convolutions
# --------------------------------------------------------------------------- #
class UnitTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, (kernel_size, 1),
                              (stride, 1), (pad, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        conv_init(self.conv)
        bn_init(self.bn)

    def forward(self, x):
        return self.bn(self.conv(x))


class MultiScaleTCN(nn.Module):
    """CTR-GCN-style multi-scale temporal conv: parallel dilated branches +
    a maxpool branch + a 1x1 branch, concatenated. Used by all models so the
    temporal modelling is identical across the comparison."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 dilations=(1, 2)):
        super().__init__()
        branch_ch = out_channels // (len(dilations) + 2)
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = (kernel_size + (kernel_size - 1) * (d - 1) - 1) // 2
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_ch, 1),
                nn.BatchNorm2d(branch_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(branch_ch, branch_ch, (kernel_size, 1),
                          (stride, 1), (pad, 0), dilation=(d, 1)),
                nn.BatchNorm2d(branch_ch),
            ))
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_ch, 1),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((3, 1), (stride, 1), (1, 0)),
            nn.BatchNorm2d(branch_ch),
        ))
        rest = out_channels - branch_ch * (len(dilations) + 1)
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, rest, 1, stride=(stride, 1)),
            nn.BatchNorm2d(rest),
        ))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], dim=1)


# --------------------------------------------------------------------------- #
# generic block + backbone
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, spatial_cls, in_channels, out_channels, A,
                 stride=1, residual=True, temporal="ms"):
        super().__init__()
        self.gcn = spatial_cls(in_channels, out_channels, A)
        if temporal == "ms":
            self.tcn = MultiScaleTCN(out_channels, out_channels, stride=stride)
        else:
            self.tcn = UnitTCN(out_channels, out_channels, stride=stride)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1,
                                    stride=stride)

    def forward(self, x):
        return self.relu(self.tcn(self.gcn(x)) + self.residual(x))


class GCNBackbone(nn.Module):
    """Standard ST-GCN/CTR-GCN topology: input BN -> 10 blocks
    (channels 64->128->256, temporal stride at blocks 5 and 8) -> GAP -> FC."""

    def __init__(self, spatial_cls, num_class, A, in_channels=3,
                 base_channels=64, num_person=1, temporal="ms", drop_out=0.0):
        super().__init__()
        self.register_buffer("A", torch.tensor(A, dtype=torch.float32))
        num_point = A.shape[-1]
        self.num_person = num_person
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        def blk(ci, co, stride=1, res=True):
            return Block(spatial_cls, ci, co, self.A, stride, res, temporal)

        self.layers = nn.ModuleList([
            blk(in_channels, c1, res=False),
            blk(c1, c1), blk(c1, c1), blk(c1, c1),
            blk(c1, c2, stride=2),
            blk(c2, c2), blk(c2, c2),
            blk(c2, c3, stride=2),
            blk(c3, c3), blk(c3, c3),
        ])
        self.fc = nn.Linear(c3, num_class)
        self.drop = nn.Dropout(drop_out)
        bn_init(self.data_bn)
        nn.init.normal_(self.fc.weight, 0, (2.0 / num_class) ** 0.5)

    def extract_feat(self, x):
        # x: [N, C, T, V, M]
        N, C, T, V, M = x.shape
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        for layer in self.layers:
            x = layer(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1).mean(3).mean(1)  # GAP over T,V and persons
        return x

    def forward(self, x):
        feat = self.extract_feat(x)
        return self.fc(self.drop(feat))
