"""Phase 2 - pose extraction (RGB video -> raw 75-joint keypoints).

Runs MediaPipe Holistic over each video and keeps 33 pose + 21 + 21 hand
landmarks (face mesh dropped). z is discarded (unreliable); channel 3 is the
confidence/visibility used later for masking. Output per clip: [T, 75, 3].

    python phase2_pose_extraction.py --videos path/to/clips --out raw/

Needs: pip install mediapipe opencv-python
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _to_arr(landmarks, n, use_visibility=False):
    out = np.zeros((n, 3), dtype=np.float32)
    if landmarks is None:
        return out                       # absent -> zeros (conf 0 => masked later)
    for i, lm in enumerate(landmarks.landmark):
        conf = lm.visibility if use_visibility else 1.0
        out[i] = (lm.x, lm.y, conf)
    return out


def _landmarks_75(res):
    pose = _to_arr(res.pose_landmarks, 33, use_visibility=True)
    lh = _to_arr(res.left_hand_landmarks, 21)
    rh = _to_arr(res.right_hand_landmarks, 21)
    return np.concatenate([pose, lh, rh], axis=0)    # [75, 3]


def extract_video(path) -> np.ndarray:
    import cv2
    import mediapipe as mp

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1, refine_face_landmarks=False)
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            res = holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(_landmarks_75(res))
    finally:
        cap.release()
        holistic.close()
    return np.stack(frames) if frames else np.zeros((0, 75, 3), np.float32)


def main():
    ap = argparse.ArgumentParser(description="Phase 2 pose extraction")
    ap.add_argument("--videos", required=True, help="folder of video clips")
    ap.add_argument("--out", required=True, help="output folder for <clip>.npy")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vids = [p for p in Path(a.videos).rglob("*") if p.suffix.lower() in VIDEO_EXT]
    print(f"{len(vids)} videos -> {out}")
    for i, v in enumerate(vids, 1):
        dst = out / f"{v.stem}.npy"
        if dst.exists():
            continue
        arr = extract_video(v)
        np.save(dst, arr)
        print(f"  [{i}/{len(vids)}] {v.name} -> {arr.shape}")


if __name__ == "__main__":
    main()
