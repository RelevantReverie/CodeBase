"""Phase 11 - evaluation & comparison metrics.

Computes Top-1, Top-5, mean per-class accuracy, and provides the signer-gap
helper. Can score a single checkpoint or fuse the 4 streams of a model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from phase4_graph import Graph
from phase5_streams import STREAMS
from phase7_dataset import make_loader, num_classes_of
from phase8_models import build_model
from phase10_fusion import fuse_scores


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


def compute_metrics(logits, labels, num_class):
    logits_t = torch.from_numpy(logits)
    labels_t = torch.from_numpy(labels)
    acc = common.topk_accuracy(logits_t, labels_t, ks=(1, 5))
    preds = logits.argmax(1)
    return {
        "top1": acc["top1"],
        "top5": acc["top5"],
        "mean_class": common.mean_class_accuracy(preds, labels, num_class),
    }


def evaluate_model(model_name, ckpt_dir, test_pkl, streams=("joint",),
                   base_channels=64, device=None):
    """Load each stream's checkpoint, collect logits, fuse, and score."""
    device = device or common.get_device()
    graph = Graph()
    num_class = num_classes_of(test_pkl)
    stream_logits, labels = {}, None
    for stream in streams:
        model = build_model(model_name, num_class, graph, base_channels=base_channels).to(device)
        ckpt = Path(ckpt_dir) / f"{model_name}_{stream}_best.pt"
        model.load_state_dict(torch.load(ckpt, map_location=device))
        loader = make_loader(test_pkl, stream, batch_size=32, train=False, parents=graph.parents)
        lg, labels = collect_logits(model, loader, device)
        stream_logits[stream] = lg
    fused = fuse_scores(stream_logits)
    return compute_metrics(fused, labels, num_class)


def main():
    ap = argparse.ArgumentParser(description="Phase 11 evaluation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--streams", nargs="*", default=["joint", "bone", "joint_motion", "bone_motion"])
    ap.add_argument("--base-channels", type=int, default=64)
    a = ap.parse_args()
    m = evaluate_model(a.model, a.ckpt_dir, a.test, a.streams, a.base_channels)
    print(f"{a.model}: top1 {m['top1']:.4f}  top5 {m['top5']:.4f}  "
          f"mean-class {m['mean_class']:.4f}")


if __name__ == "__main__":
    main()
