"""InfoGCN (Chi et al., CVPR 2022) spatial module.

Self-Attention Graph Convolution (SA-GC): the topology is data-dependent - an
attention adjacency inferred from joint features is added to a learnable base
topology. (InfoGCN's information-bottleneck loss term is left out so every model
trains under the identical loss; it can be added back in Phase 9 if wanted.)
"""
import torch
import torch.nn as nn

from .base import GCNBackbone, conv_init, bn_init


class SAGCSpatial(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        A = A if torch.is_tensor(A) else torch.as_tensor(A, dtype=torch.float32)
        self.K = A.shape[0]
        self.out_channels = out_channels
        self.PA = nn.Parameter(A.clone().float())
        self.q = nn.Conv2d(in_channels, out_channels, 1)
        self.k = nn.Conv2d(in_channels, out_channels, 1)
        self.v = nn.Conv2d(in_channels, out_channels * self.K, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        bn_init(self.bn)

    def forward(self, x):
        N, C, T, V = x.shape
        q = self.q(x).mean(-2)                       # N, out, V
        k = self.k(x).mean(-2)                        # N, out, V
        att = torch.softmax(torch.einsum("ncv,ncw->nvw", q, k) / (q.shape[1] ** 0.5), dim=-1)
        v = self.v(x).view(N, self.K, self.out_channels, T, V)
        out = 0
        for i in range(self.K):
            A_eff = self.PA[i].unsqueeze(0) + self.alpha * att   # N, V, V
            out = out + torch.einsum("nctv,nvw->nctw", v[:, i], A_eff)
        return self.bn(out)


def build_model(num_class, graph, **kw):
    return GCNBackbone(SAGCSpatial, num_class, graph.A, **kw)
