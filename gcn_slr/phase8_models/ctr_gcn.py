"""CTR-GCN (Chen et al., ICCV 2021) spatial module.

Channel-wise Topology Refinement: a shared learnable topology per subset plus a
per-channel refinement inferred from pairwise joint feature differences.
"""
import torch
import torch.nn as nn

from .base import GCNBackbone, conv_init, bn_init


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8):
        super().__init__()
        mid = 8 if in_channels <= 3 else out_channels // rel_reduction
        self.conv1 = nn.Conv2d(in_channels, mid, 1)
        self.conv2 = nn.Conv2d(in_channels, mid, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv4 = nn.Conv2d(mid, out_channels, 1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)

    def forward(self, x, A, alpha):
        x1 = self.conv1(x).mean(-2)          # N, mid, V
        x2 = self.conv2(x).mean(-2)          # N, mid, V
        x3 = self.conv3(x)                   # N, out, T, V
        diff = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))   # N, mid, V, V
        a = self.conv4(diff) * alpha + A.unsqueeze(0).unsqueeze(0)  # N, out, V, V
        return torch.einsum("ncuv,nctv->nctu", a, x3)


class CTRSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        A = A if torch.is_tensor(A) else torch.as_tensor(A, dtype=torch.float32)
        self.K = A.shape[0]
        self.convs = nn.ModuleList([CTRGC(in_channels, out_channels) for _ in range(self.K)])
        self.PA = nn.Parameter(A.clone().float())
        self.alpha = nn.Parameter(torch.zeros(1))
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                      nn.BatchNorm2d(out_channels))
        else:
            self.down = nn.Identity()
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        out = 0
        for k, conv in enumerate(self.convs):
            out = out + conv(x, self.PA[k], self.alpha)
        out = self.bn(out) + self.down(x)
        return self.relu(out)


def build_model(num_class, graph, **kw):
    return GCNBackbone(CTRSpatial, num_class, graph.A, **kw)
