"""Phase 9 - training (one model, one stream). Same recipe for all six models.

    python phase9_train.py --model ctr_gcn --stream joint \
        --train processed/ISL/train.pkl --val processed/ISL/val.pkl --out runs/ISL

Loop over MODEL_NAMES x STREAMS to produce every checkpoint for the comparison.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from phase4_graph import Graph
from phase6_augment import Augmentor
from phase7_dataset import make_loader, num_classes_of
from phase8_models import build_model, MODEL_NAMES

DEFAULTS = dict(
    epochs=100, batch_size=32, lr=0.1, weight_decay=5e-4, warmup=5,
    label_smoothing=0.1, base_channels=64, seed=1,
)


def cosine_warmup(optimizer, warmup, total):
    def f(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        p = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


@torch.no_grad()
def eval_top1(model, loader, device):
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        n += len(y)
    return correct / max(n, 1)


def train_one(model_name, cfg, train_pkl, val_pkl, out_dir, stream="joint", device=None):
    device = device or common.get_device()
    common.set_seed(cfg["seed"])
    graph = Graph()
    num_class = cfg.get("num_class") or num_classes_of(train_pkl)
    model = build_model(model_name, num_class, graph,
                        base_channels=cfg["base_channels"]).to(device)

    aug = Augmentor(cfg.get("augment"))
    tl = make_loader(train_pkl, stream, cfg["batch_size"], train=True,
                     augment=aug, parents=graph.parents)
    vl = make_loader(val_pkl, stream, cfg["batch_size"], train=False, parents=graph.parents)

    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9,
                          nesterov=True, weight_decay=cfg["weight_decay"])
    sched = cosine_warmup(opt, cfg["warmup"], cfg["epochs"])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(cfg["epochs"]):
        model.train()
        meter = common.AverageMeter()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            meter.update(loss.item(), len(y))
        sched.step()
        acc = eval_top1(model, vl, device)
        print(f"[{model_name}/{stream}] ep {ep+1}/{cfg['epochs']} "
              f"loss {meter.avg:.3f} val@1 {acc:.4f}")
        if acc >= best:
            best = acc
            torch.save(model.state_dict(), Path(out_dir) / f"{model_name}_{stream}_best.pt")
    print(f"[{model_name}/{stream}] best val@1 {best:.4f}")
    return best


def main():
    ap = argparse.ArgumentParser(description="Phase 9 training")
    ap.add_argument("--model", required=True, choices=MODEL_NAMES)
    ap.add_argument("--stream", default="joint")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-class", type=int, default=None)
    a = ap.parse_args()

    cfg = dict(DEFAULTS)
    if a.config:
        cfg = common.merge(cfg, common.load_config(a.config))
    if a.epochs is not None:
        cfg["epochs"] = a.epochs
    if a.num_class is not None:
        cfg["num_class"] = a.num_class
    train_one(a.model, cfg, a.train, a.val, a.out, a.stream)


if __name__ == "__main__":
    main()
