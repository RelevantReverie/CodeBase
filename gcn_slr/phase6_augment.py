"""Phase 6 - train-time augmentation (applied to the joint sample before Phase 5).

Ranges simulate an unseen signer / camera without changing the sign's identity
(see pipeline doc). Operates on x: [C, T, V, M] numpy, transforming only the x,y
channels (0,1); the confidence channel (2) is left untouched. NO horizontal flip
by default - it swaps handedness and can change a sign's meaning.
"""
import numpy as np

DEFAULTS = dict(
    rotation=13.0,      # degrees
    scale=0.1,          # +/- 10%
    translation=0.05,   # +/- 5%
    shear=0.1,
    speed=0.2,          # +/- 20%
    frame_drop=0.1,     # up to 10% of frames
    joint_mask=3,       # up to 3 joints
    noise=0.015,        # gaussian sigma
)


class Augmentor:
    def __init__(self, cfg=None):
        self.c = {**DEFAULTS, **(cfg or {})}

    def __call__(self, x):
        x = x.copy()
        xy = x[:2]                                   # [2, T, V, M]
        C2, T, V, M = xy.shape

        # rotation
        if self.c["rotation"]:
            a = np.deg2rad(np.random.uniform(-self.c["rotation"], self.c["rotation"]))
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
            xy = np.einsum("ij,jtvm->itvm", R, xy)
        # scale
        if self.c["scale"]:
            xy *= np.random.uniform(1 - self.c["scale"], 1 + self.c["scale"])
        # shear
        if self.c["shear"]:
            s1, s2 = np.random.uniform(-self.c["shear"], self.c["shear"], 2)
            S = np.array([[1, s1], [s2, 1]], np.float32)
            xy = np.einsum("ij,jtvm->itvm", S, xy)
        # translation
        if self.c["translation"]:
            off = np.random.uniform(-self.c["translation"], self.c["translation"], (2, 1, 1, 1))
            xy = xy + off.astype(np.float32)
        # coordinate noise
        if self.c["noise"]:
            xy = xy + np.random.normal(0, self.c["noise"], xy.shape).astype(np.float32)

        x[:2] = xy

        # temporal speed (resample then restore length)
        if self.c["speed"]:
            f = np.random.uniform(1 - self.c["speed"], 1 + self.c["speed"])
            newT = max(2, int(round(T * f)))
            idx = np.linspace(0, T - 1, newT).round().astype(int)
            x = x[:, idx]
            idx2 = np.linspace(0, newT - 1, T).round().astype(int)
            x = x[:, idx2]
        # frame drop
        if self.c["frame_drop"]:
            k = int(self.c["frame_drop"] * x.shape[1])
            if k:
                drop = np.random.choice(x.shape[1], k, replace=False)
                x[:, drop] = 0
        # joint mask
        if self.c["joint_mask"]:
            k = np.random.randint(0, self.c["joint_mask"] + 1)
            if k:
                jm = np.random.choice(V, k, replace=False)
                x[:, :, jm, :] = 0
        return x.astype(np.float32)
