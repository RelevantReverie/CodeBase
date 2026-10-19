"""Phase 5 - multi-stream generation.

From the joint tensor derive the four streams every GCN is evaluated on:
    joint, bone, joint_motion, bone_motion
Operates on a single sample [C, T, V, M] (numpy). The bone stream uses the
parent map from Phase 4 (translation-invariant by construction).
"""
import numpy as np

STREAMS = ["joint", "bone", "joint_motion", "bone_motion"]


def make_bone(x, parents):
    # x: [C, T, V, M]; bone[v] = joint[v] - joint[parent[v]]
    bone = np.zeros_like(x)
    for v in range(x.shape[2]):
        bone[:, :, v, :] = x[:, :, v, :] - x[:, :, parents[v], :]
    return bone


def make_motion(x):
    m = np.zeros_like(x)
    m[:, :-1] = x[:, 1:] - x[:, :-1]
    return m


def to_stream(x, stream, parents):
    if stream == "joint":
        return x
    if stream == "bone":
        return make_bone(x, parents)
    if stream == "joint_motion":
        return make_motion(x)
    if stream == "bone_motion":
        return make_motion(make_bone(x, parents))
    raise ValueError(f"unknown stream {stream}; choose from {STREAMS}")
