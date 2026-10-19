"""Shared utilities: config loading, seeding, device, metrics.

Used by the training / eval phases. Kept dependency-light (PyYAML optional).
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # torch optional for pure-data phases
    torch = None


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "cuda"):
    if torch is not None and prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if torch is not None else "cpu"


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path) -> dict:
    """Load a YAML (preferred) or JSON config. Falls back to a tiny YAML
    parser if PyYAML is absent so the package runs anywhere."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
            return yaml.safe_load(text)
        except Exception:
            return _mini_yaml(text)
    return json.loads(text)


def merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def _mini_yaml(text: str):
    """Minimal flat/one-level YAML reader (key: value, 2-space nesting).
    Enough for the simple configs shipped here; install PyYAML for full YAML."""
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        key, _, val = raw.strip().partition(":")
        val = val.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if val == "":
            d: dict = {}
            parent[key] = d
            stack.append((indent, d))
        else:
            parent[key] = _coerce(val)
    return root


def _coerce(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("null", "none", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_coerce(x.strip()) for x in inner.split(",")] if inner else []
    return v.strip('"\'')


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, val, k=1):
        self.sum += float(val) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


def topk_accuracy(logits, labels, ks=(1, 5)):
    """Top-k accuracy. logits: [N, num_class] tensor; labels: [N]."""
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)           # [N, maxk]
    correct = pred.eq(labels.view(-1, 1))
    out = {}
    for k in ks:
        kk = min(k, maxk)
        out[f"top{k}"] = correct[:, :kk].any(dim=1).float().mean().item()
    return out


def mean_class_accuracy(preds, labels, num_classes):
    """Macro (mean per-class) recall — the imbalance-aware metric."""
    preds = np.asarray(preds)
    labels = np.asarray(labels)
    accs = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        accs.append((preds[mask] == c).mean())
    return float(np.mean(accs)) if accs else 0.0


def signer_gap(signer_dependent_acc: float, signer_independent_acc: float) -> float:
    """Direct measure of signer invariance: how much accuracy drops when
    moving from seen to unseen signers. Lower is more invariant."""
    return signer_dependent_acc - signer_independent_acc
