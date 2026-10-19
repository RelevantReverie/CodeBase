"""End-to-end smoke test on SYNTHETIC data - verifies every phase wires together
without needing the real datasets or a GPU. Run:  python smoke_test.py
"""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from phase3_preprocess import preprocess_clip
from phase4_graph import Graph
from phase5_streams import STREAMS, to_stream
from phase6_augment import Augmentor
from phase10_fusion import fuse_scores
from phase11_eval import compute_metrics

NUM_CLASS, V, T = 8, 75, 64


def section(msg):
    print("\n==", msg)


def test_data_phases(graph):
    section("Phase 3: preprocess a random raw clip")
    raw = np.random.rand(37, V, 3).astype(np.float32)   # [T_raw, 75, 3]
    raw[:, :, 2] = 0.9                                   # high confidence
    arr = preprocess_clip(raw)
    assert arr.shape == (3, T, V, 1), arr.shape
    print("   ok ->", arr.shape)

    section("Phase 6 + 5: augment then build all 4 streams")
    aug = Augmentor()
    a = aug(arr)
    assert a.shape == arr.shape
    for s in STREAMS:
        st = to_stream(a, s, graph.parents)
        assert st.shape == (3, T, V, 1), (s, st.shape)
    print("   ok ->", STREAMS)


def test_models(graph):
    import torch
    section("Phase 8: forward + backward for every model")
    x = torch.randn(2, 3, T, V, 1)
    y = torch.randint(0, NUM_CLASS, (2,))
    from phase8_models import build_model, MODEL_NAMES
    for name in MODEL_NAMES:
        model = build_model(name, NUM_CLASS, graph, base_channels=16)
        out = model(x)
        assert out.shape == (2, NUM_CLASS), (name, out.shape)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   {name:9s} ok  logits {tuple(out.shape)}  params {n_params/1e6:.2f}M")


def _make_pkl(path, n):
    data = np.random.randn(n, 3, T, V, 1).astype(np.float32)
    label = np.random.randint(0, NUM_CLASS, n).astype(np.int64)
    signer = [f"s{i%3}" for i in range(n)]
    with open(path, "wb") as f:
        pickle.dump({"data": data, "label": label, "signer": signer}, f)


def test_train_eval(graph):
    import torch  # noqa
    from phase9_train import train_one, DEFAULTS
    from phase11_eval import collect_logits
    from phase7_dataset import make_loader
    section("Phase 9 + 11 + 10: mini train, eval, fusion")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _make_pkl(d / "train.pkl", 16)
        _make_pkl(d / "val.pkl", 8)
        _make_pkl(d / "test.pkl", 8)
        cfg = dict(DEFAULTS, epochs=1, batch_size=8, base_channels=16, num_class=NUM_CLASS)
        train_one("ctr_gcn", cfg, d / "train.pkl", d / "val.pkl", d / "run", stream="joint",
                  device=common.get_device("cpu"))

        # eval + fusion over two streams using the freshly trained joint ckpt for both
        from phase8_models import build_model
        device = common.get_device("cpu")
        stream_logits, labels = {}, None
        for stream in ("joint", "bone"):
            m = build_model("ctr_gcn", NUM_CLASS, graph, base_channels=16).to(device)
            m.load_state_dict(torch.load(d / "run" / "ctr_gcn_joint_best.pt", map_location=device))
            loader = make_loader(d / "test.pkl", stream, batch_size=8, parents=graph.parents)
            lg, labels = collect_logits(m, loader, device)
            stream_logits[stream] = lg
        fused = fuse_scores(stream_logits)
        mets = compute_metrics(fused, labels, NUM_CLASS)
        print(f"   fused metrics: {mets}")


def main():
    common.set_seed(0)
    graph = Graph()
    section("Phase 4: graph")
    print(f"   A {graph.A.shape}  A_left {graph.A_left.shape}  parents {graph.parents.shape}")
    test_data_phases(graph)
    try:
        import torch  # noqa
    except Exception:
        print("\n[torch not installed -> skipping model/train/eval phases]")
        print("\nSMOKE TEST (data phases) PASSED")
        return
    test_models(graph)
    test_train_eval(graph)
    print("\nSMOKE TEST PASSED - every phase wired correctly.")


if __name__ == "__main__":
    main()
