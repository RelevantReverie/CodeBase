"""Phase 3 - keypoint preprocessing & normalization (the signer-invariance core).

Raw per-clip landmarks [T, 75, 3] (x, y, confidence) -> canonical tensor
[C=3, T=64, V=75, M=1]. Every value below is justified in GCN_Training_Pipeline.md.

Pipeline: confidence mask -> short-gap interpolation -> mid-shoulder centering
-> shoulder-width scaling -> shoulder-line rotation -> temporal resample to T.
"""
from __future__ import annotations

import numpy as np

# ---- fixed parameters (see pipeline doc for the rationale of each) ---------- #
CONF_TH = 0.5        # below this a landmark is a guess -> treat as missing
MAX_GAP = 7          # interpolate occlusions <= ~0.25s (@30fps); longer -> zero
T_OUT = 64           # fixed clip length: covers a full isolated sign
LSHO, RSHO = 11, 12  # MediaPipe pose shoulder indices (the normalization anchors)


def _fill_gaps(xy, valid):
    """Linear-interpolate short gaps per joint; zero gaps longer than MAX_GAP."""
    T, V, _ = xy.shape
    out = xy.copy()
    for v in range(V):
        col = valid[:, v]
        if not col.any():
            out[:, v, :] = 0
            continue
        idx = np.where(col)[0]
        for c in range(2):
            out[:, v, c] = np.interp(np.arange(T), idx, out[idx, v, c])
        # re-zero long invalid runs (incl. long leading/trailing extrapolation)
        invalid = ~col
        i = 0
        while i < T:
            if invalid[i]:
                j = i
                while j < T and invalid[j]:
                    j += 1
                if (j - i) > MAX_GAP:
                    out[i:j, v, :] = 0
                i = j
            else:
                i += 1
    return out


def preprocess_clip(raw) -> np.ndarray:
    """raw [T, 75, 3] -> [3, 64, 75, 1]."""
    raw = np.asarray(raw, dtype=np.float32)
    T, V, _ = raw.shape
    xy = raw[:, :, :2].copy()
    conf = raw[:, :, 2].copy()

    # 3.2 missing-data handling
    valid = conf >= CONF_TH
    xy = _fill_gaps(xy, valid)

    # 3.3 translation: center on the mid-shoulder point (per frame)
    center = (xy[:, LSHO, :] + xy[:, RSHO, :]) / 2.0        # [T, 2]
    xy = xy - center[:, None, :]

    # 3.4 scale: divide by shoulder width (per-clip median constant)
    sw = np.linalg.norm(xy[:, LSHO, :] - xy[:, RSHO, :], axis=1)
    sw = sw[sw > 1e-6]
    scale = float(np.median(sw)) if sw.size else 1.0
    scale = scale if scale > 1e-6 else 1.0
    xy = xy / scale

    # 3.5 rotation: align the shoulder line to the horizontal
    vec = (xy[:, RSHO, :] - xy[:, LSHO, :]).mean(0)         # [2]
    ang = np.arctan2(vec[1], vec[0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    xy = xy @ R.T

    # 3.6 temporal resample to T_OUT (uniform sampling / loop)
    idx = np.linspace(0, max(T - 1, 0), T_OUT).round().astype(int)
    xy = xy[idx]
    conf = conf[idx]

    # 3.7 assemble [C=3, T, V, M=1]  (channels = x, y, confidence)
    out = np.concatenate([xy, conf[:, :, None]], axis=2)   # [T, V, 3]
    return out.transpose(2, 0, 1)[:, :, :, None].astype(np.float32)


# --------------------------------------------------------------------------- #
# CLI: turn a folder of raw-keypoint .npy + a manifest into processed splits
# --------------------------------------------------------------------------- #
def process_manifest(raw_dir, manifest_csv, out_dir):
    """manifest_csv columns: clip_id,label,signer,split
    raw_dir/<clip_id>.npy holds [T,75,3]. Writes out_dir/<split>.pkl."""
    import csv
    import pickle
    from collections import defaultdict
    from pathlib import Path

    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    buckets = defaultdict(lambda: {"data": [], "label": [], "signer": [], "clip_id": []})

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            raw = np.load(raw_dir / f"{row['clip_id']}.npy")
            arr = preprocess_clip(raw)
            b = buckets[row["split"]]
            b["data"].append(arr)
            b["label"].append(int(row["label"]))
            b["signer"].append(row["signer"])
            b["clip_id"].append(row["clip_id"])

    for split, b in buckets.items():
        b["data"] = np.stack(b["data"]).astype(np.float32)
        b["label"] = np.asarray(b["label"], dtype=np.int64)
        with open(out_dir / f"{split}.pkl", "wb") as f:
            pickle.dump(b, f)
        print(f"{split}: {b['data'].shape} -> {out_dir / f'{split}.pkl'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Phase 3 preprocessing")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    process_manifest(a.raw_dir, a.manifest, a.out_dir)
