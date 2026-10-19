"""Phase 4 - Skeleton graph for the 75-joint (hands + upper body) layout.

Layout (V = 75):
    pose   : indices  0..32   (MediaPipe Pose, 33 landmarks)
    L hand : indices 33..53   (MediaPipe Hand, 21 landmarks, local 0..20)
    R hand : indices 54..74   (MediaPipe Hand, 21 landmarks, local 0..20)

Provides:
    Graph().A            -> [3, 75, 75]  spatial-partition adjacency (self/inward/outward)
    Graph().A_left/right -> [3, 75, 75]  adjacency restricted to one hand (for HA-GCN)
    Graph().parents      -> [75]         bone parents (for the bone stream, Phase 5)
    LEFT_HAND / RIGHT_HAND index lists
"""
from __future__ import annotations

import numpy as np

NUM_NODE = 75
POSE = list(range(0, 33))
LEFT_HAND = list(range(33, 54))   # global ids of the 21 left-hand joints
RIGHT_HAND = list(range(54, 75))  # global ids of the 21 right-hand joints

# MediaPipe hand topology, local joint ids 0..20 (0 = wrist)
_HAND_LOCAL = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm edge
]

# MediaPipe pose edges we keep (upper body + face-light + legs for completeness)
_POSE_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),        # left face
    (0, 4), (4, 5), (5, 6), (6, 8),        # right face
    (9, 10),                               # mouth
    (0, 11), (0, 12),                      # link face to shoulders (graph connectivity)
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms
    (11, 23), (12, 24), (23, 24),          # torso
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # pose-hand stubs
    (23, 25), (25, 27), (27, 29), (29, 31),  # left leg (usually cropped/masked)
    (24, 26), (26, 28), (28, 30), (30, 32),  # right leg
]

ROOT = 11  # mid-upper-body anchor (left shoulder) for the bone tree


def _hand_edges(offset):
    return [(a + offset, b + offset) for a, b in _HAND_LOCAL]


def _all_edges():
    edges = list(_POSE_EDGES)
    edges += _hand_edges(33)                     # left hand internal
    edges += _hand_edges(54)                     # right hand internal
    edges += [(15, 33), (16, 54)]                # pose wrist -> hand wrist (cross-links)
    return edges


def _bfs_parents(edges, root=ROOT, num_node=NUM_NODE):
    adj = {i: [] for i in range(num_node)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    parents = [root] * num_node
    seen = {root}
    queue = [root]
    while queue:
        u = queue.pop(0)
        for v in sorted(adj[u]):
            if v not in seen:
                seen.add(v)
                parents[v] = u
                queue.append(v)
    return parents


# --- adjacency helpers (ST-GCN / CTR-GCN style spatial partitioning) --------- #
def _edge2mat(link, num_node=NUM_NODE):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def _normalize_digraph(A):
    Dl = A.sum(0)
    n = A.shape[0]
    Dn = np.zeros((n, n))
    for i in range(n):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return A @ Dn


def _spatial_graph(inward, num_node=NUM_NODE):
    """Return [3, V, V]: self-link, inward (normalized), outward (normalized)."""
    self_mat = np.eye(num_node)
    outward = [(j, i) for i, j in inward]
    In = _normalize_digraph(_edge2mat(inward, num_node))
    Out = _normalize_digraph(_edge2mat(outward, num_node))
    return np.stack((self_mat, In, Out)).astype(np.float32)


class Graph:
    """Holds the adjacency tensors and bone parents for the 75-joint skeleton."""

    def __init__(self):
        edges = _all_edges()
        self.parents = np.array(_bfs_parents(edges), dtype=np.int64)

        # inward edges = (child, parent) from the BFS tree
        inward = [(v, int(self.parents[v])) for v in range(NUM_NODE) if v != ROOT]
        self.A = _spatial_graph(inward)

        # per-hand adjacency: keep only inward edges whose child is in that hand
        left_in = [(c, p) for (c, p) in inward if c in LEFT_HAND]
        right_in = [(c, p) for (c, p) in inward if c in RIGHT_HAND]
        self.A_left = _spatial_graph(left_in)
        self.A_right = _spatial_graph(right_in)

    def num_node(self):
        return NUM_NODE


if __name__ == "__main__":
    g = Graph()
    print("A:", g.A.shape, "A_left:", g.A_left.shape, "parents:", g.parents.shape)
    print("edges:", len(_all_edges()), "root:", ROOT)
    assert g.A.shape == (3, 75, 75)
    assert g.parents.shape == (75,)
    print("Graph OK")
