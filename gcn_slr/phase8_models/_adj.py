"""Adjacency helpers shared by the multi-scale / hierarchical models."""
import numpy as np


def to_numpy_binary(A):
    """A can be [3,V,V] torch/np -> binary [V,V] numpy."""
    import numpy as _np
    try:
        A = A.detach().cpu().numpy()
    except Exception:
        A = _np.asarray(A)
    if A.ndim == 3:
        A = A.sum(0)
    return (A > 0).astype(np.float32)


def k_adjacency(A_bin, k):
    """Disentangled k-hop adjacency (MS-G3D style): exactly-k-hop neighbours."""
    I = np.eye(A_bin.shape[0], dtype=np.float32)
    if k == 0:
        return I
    Ak = (np.linalg.matrix_power(A_bin + I, k) > 0).astype(np.float32)
    Akm1 = (np.linalg.matrix_power(A_bin + I, k - 1) > 0).astype(np.float32)
    return Ak - Akm1


def normalize_adj(A):
    """Symmetric normalization D^-1/2 (A+I) D^-1/2."""
    A = A + np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(-1)
    dinv = np.zeros_like(d)
    dinv[d > 0] = d[d > 0] ** -0.5
    D = np.diag(dinv)
    return (D @ A @ D).astype(np.float32)
