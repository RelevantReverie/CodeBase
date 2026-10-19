"""Phase 10 - multi-stream late fusion (part of getting the best model).

Weighted softmax sum of the 4 streams. Default weights
[joint 1, bone 1, joint_motion 0.5, bone_motion 0.5] (tune on val).
"""
from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {"joint": 1.0, "bone": 1.0, "joint_motion": 0.5, "bone_motion": 0.5}


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fuse_scores(stream_logits: dict, weights: dict | None = None):
    """stream_logits: {stream_name: [N, num_class] logits}. Returns fused [N, num_class]."""
    weights = weights or DEFAULT_WEIGHTS
    fused = None
    for name, logits in stream_logits.items():
        w = weights.get(name, 1.0)
        probs = softmax(np.asarray(logits)) * w
        fused = probs if fused is None else fused + probs
    return fused


def fuse_predict(stream_logits: dict, weights: dict | None = None):
    return fuse_scores(stream_logits, weights).argmax(1)
