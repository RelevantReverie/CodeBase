# gcn_slr - skeleton GCN pipeline for sign-language recognition

Self-contained PyTorch code for the pipeline in `../GCN_Training_Pipeline.md`.
One module per phase (shared, model-agnostic) + all six GCNs behind one registry.
Input everywhere is the 75-joint skeleton tensor `[N, C=3, T=64, V=75, M=1]`.

```
gcn_slr/
├── common.py                 # seeds, config, metrics (incl. signer_gap)
├── phase2_pose_extraction.py # RGB video -> raw [T,75,3]   (MediaPipe)
├── phase3_preprocess.py      # raw -> normalized [3,64,75,1] (signer-invariance core)
├── phase4_graph.py           # 75-joint graph + hand sub-graphs + bone parents
├── phase5_streams.py         # joint / bone / joint_motion / bone_motion
├── phase6_augment.py         # train-time augmentation
├── phase7_dataset.py         # Dataset, signer-independent splits, loaders
├── phase8_models/            # ST-GCN, MS-G3D, CTR-GCN, InfoGCN, HD-GCN, HA-GCN
│   ├── __init__.py           #   build_model(name, num_class, graph)
│   └── base.py               #   shared blocks + 10-block backbone
├── phase9_train.py           # training (one model, one stream)
├── phase10_fusion.py         # 4-stream late fusion
├── phase11_eval.py           # Top-1/5, mean-class, signer-gap
├── configs/                  # default.yaml + per-model yamls
└── smoke_test.py             # synthetic end-to-end check
```

## Quick start

```bash
pip install -r requirements.txt
python smoke_test.py          # verifies all phases wire together (CPU, synthetic data)
```

## Real run (per dataset)

```bash
# Phase 2: videos -> raw keypoints
python phase2_pose_extraction.py --videos data/ISL/videos --out data/ISL/raw

# Phase 3: raw + manifest(clip_id,label,signer,split) -> processed splits
python phase3_preprocess.py --raw-dir data/ISL/raw --manifest data/ISL/manifest.csv --out-dir processed/ISL

# Phase 9: train every model x every stream (loop in your shell)
for m in st_gcn ms_g3d ctr_gcn info_gcn hd_gcn ha_gcn; do
  for s in joint bone joint_motion bone_motion; do
    python phase9_train.py --model $m --stream $s \
      --train processed/ISL/train.pkl --val processed/ISL/val.pkl \
      --config configs/$m.yaml --out runs/ISL
  done
done

# Phase 11: fuse the 4 streams and score on the signer-independent test set
python phase11_eval.py --model ha_gcn --ckpt-dir runs/ISL --test processed/ISL/test.pkl
```

## Design notes (what to know before trusting results)

- **Shared phases + per-model configs** - phases 2-7 are identical for all six; only
  `phase8_models/<model>.py` differs. Same graph, same tensor, same training budget,
  same streams -> the comparison reflects the architecture, not the setup.
- **Faithfulness:** ST-GCN and CTR-GCN are close to source. MS-G3D, InfoGCN, HD-GCN and
  HA-GCN implement each paper's *core* idea but are reconstructions (MS-G3D's G3D window,
  InfoGCN's IB loss, and HA-GCN's exact DropGraph are simplified - noted in each file).
  For published-accuracy reproduction, swap in the official repo behind the same
  `build_model` interface.
- **Signer invariance** is the objective: 75 joints, drop-z, mid-shoulder centering,
  shoulder-width scaling, shoulder-line rotation, speed augmentation, bone/motion
  streams, signer-independent splits, and the **signer-gap** metric in `phase11_eval`.
- All models use the shared multi-scale temporal conv (so ST-GCN here differs from its
  original single-scale TCN) - a deliberate fairness choice.
```
