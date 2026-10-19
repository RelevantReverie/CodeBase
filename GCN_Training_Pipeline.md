# Training & Comparing 6 Skeleton GCNs — Detailed Pipeline

**Models:** ST-GCN · MS-G3D · CTR-GCN · InfoGCN · HD-GCN · **HA-GCN** (Hand-Aware GCN, Song et al. 2024)
**Task:** Isolated Sign Language Recognition (ISLR) — one clip → one gloss, across ISL / ASL / BSL / Auslan / Turkish
**Primary objective:** maximise accuracy **on unseen signers** — i.e. a **signer-invariant** model — and compare the six fairly.

Every phase below is written as **What → How (the exact value) → Why that value → Signer-invariance contribution**, because the whole study rises or falls on (a) making the input signer-invariant and (b) changing *only* the model between runs.

---

## 0. Governing principles

1. **Hold everything constant except the network.** Pose backend, joint layout, graph, tensor shape, splits, normalization, augmentation, training budget, fusion, seeds, and metrics are frozen once and reused for all six models. Otherwise a "win" is just a config difference.
2. **Signer invariance is an explicit design target, not a side effect.** Pose (not RGB) already removes appearance/clothing/skin/background — the biggest signer confounds. Everything else below removes the *geometric* signer differences: body size, position, camera distance, orientation, and signing speed.

---

## Phase 0 — Frozen decisions

| Decision | Choice | Why this choice |
|---|---|---|
| Pose backend | **MediaPipe Holistic** (or RTMPose WholeBody) | Runs on any machine, gives body+both hands jointly, well-tested for SLR. |
| Joint layout **V** | **75** = 33 pose + 21 left hand + 21 right hand (face mesh dropped) | Hands carry the lexical content (handshape/orientation/motion); upper body gives arm trajectory + signing-space location; face mesh (468 pts) is mostly noise for manual signs and explodes the graph. 75 is an established SLR layout. |
| Channels **C** | **(x, y, confidence)** → C=3; **z dropped** | MediaPipe's z (depth) is documented as unreliable and varies with camera/signer, so it would *inject* signer variance. Confidence is kept to mask bad joints. |
| Clip length **T** | **64 frames** (uniform temporal sampling) | Isolated signs are ~0.5–2.5 s; at 25–30 fps that is ~15–75 frames, so 64 captures almost all of a sign while bounding compute and pooling cleanly. Tunable (see §Parameter table). |
| Tensor contract | NTU-style **[N, C, T, V, M]**, M=1 | All six repos consume this exact shape. |
| Splits | **Signer-independent**; leave-one-signer-out where feasible | The only way to *measure* signer invariance (see Phase 11). |
| Framework | PySKL / MMAction2 for ST-GCN·CTR-GCN·MS-G3D; official repos for InfoGCN·HD-GCN·HA-GCN, all reading the same `.pkl` | Minimises code differences between models. |
| Reproducibility | fixed seeds, pinned versions, logged configs | So results are re-runnable and the seed-variance (Phase 11) is honest. |

> **Note on the 75 layout:** MediaPipe Pose's indices 23–32 (hips, knees, ankles, feet — 10 points) are usually out of frame in SLR. We keep the 75-node graph fixed but these nodes are confidence-masked to near-zero, so they add no signal and no signer noise. If you prefer, prune them to a tight **65-node** graph — just keep it identical across all six models.

---

## Phase 1 — Data acquisition  *(done — `ISLR/` scripts)*

Download each dataset to its own folder, verify file counts/checksums, record the dataset version, and build a manifest: `clip_id, label, signer_id, split`. The **`signer_id` is mandatory** — without it you cannot build signer-independent splits or measure the signer gap.

---

## Phase 2 — Pose extraction (RGB → raw keypoints)

**What:** run the pose backend frame-by-frame to get raw landmarks per clip.
**How:** MediaPipe Holistic → per frame `(33 pose + 21 + 21) × (x, y, visibility)`; keep x,y in normalized image coords [0,1], keep visibility as confidence; **discard z**. Store one `.npy` per clip (`[T_raw, 75, 3]`).
**Why these values:** normalized image coords make the next step resolution-independent; dropping z removes an unreliable, signer/camera-dependent axis.
**Signer-invariance contribution:** switching from pixels to 75 landmarks already discards clothing, skin tone, lighting, and background — variance that has nothing to do with the sign.

**Critical detail — re-extract the pose-shipping datasets too.** iSign, MM-WLAuslan, BosphorusSign22k and AUTSL ship pose, but from *different* extractors (OpenPose / Kinect). Feeding mixed conventions is a hidden signer/domain confound; run every dataset through the **same** backend (or remap carefully) so all 75-joint tensors mean the same thing.

---

## Phase 3 — Keypoint preprocessing & normalization  *(the signer-invariance core)*

Apply in this order. Each step removes one axis of signer/camera variation.

### 3.1 Joint selection
Reduce to the fixed 75 indices (same for every dataset).

### 3.2 Missing-data handling
- **Confidence mask:** joints with visibility **< 0.5** are marked missing. *Why 0.5:* below it MediaPipe is essentially guessing; propagating a guess adds noise and, because different signers occlude differently, signer-specific noise.
- **Interpolation:** linearly interpolate a joint's gap **if ≤ 7 frames (~0.25 s)**; longer gaps stay zero-filled + masked. *Why 7 frames:* short occlusions (a hand briefly crossing the body) are smooth and recoverable; a hand leaving frame for longer should not be hallucinated.
- **Zero-fill** remaining missing joints (and keep the mask). *Why:* preserves the fixed `[T,75,3]` shape; zeros + mask tell the model "absent", a standard SLR practice.

### 3.3 Spatial normalization — translation
- **Center every frame** on the **mid-shoulder point** = midpoint of pose L-shoulder (11) and R-shoulder (12). Subtract it from all 75 joints, per frame.
- *Why mid-shoulder (not nose, not hip):* it is **always visible** in upper-body framing, **central** to the signing space, **stable** (unlike the nose, which moves with head gestures) and **present even when the lower body is cropped** (unlike the hip that NTU uses). 
- *Why per-frame:* removes global body translation — the signer swaying, standing left/right, or a panning camera. This is the single biggest **position**-invariance step.
**Signer-invariance:** kills "where the signer stands / where the camera points."

### 3.4 Spatial normalization — scale
- **Divide all coordinates by the signer's shoulder width** = ‖joint 11 − joint 12‖, computed **once per clip as the median over valid frames** and applied as a constant.
- *Why shoulder width:* the most **stable, always-visible anthropometric constant** in upper-body video; it is invariant to how close the signer is to the camera. Torso length (shoulder→hip) is better in theory but the hip is usually cropped, so shoulder width wins for SLR.
- *Why a per-clip constant (not per-frame):* a per-frame divisor would ride on shoulder-detection jitter and **distort the motion dynamics**; a constant removes body size while preserving how the sign actually moves.
**Signer-invariance:** kills "big signer vs small signer" and "near camera vs far."

### 3.5 Spatial normalization — rotation (optional but recommended)
- Rotate each frame so the shoulder line is **horizontal** (align 11–12 to the x-axis).
- *Why:* removes small camera-tilt / lean differences between recordings.
**Signer-invariance:** kills "camera/body tilt."

### 3.6 Temporal normalization
- Resample each clip to **T = 64** by **uniform sampling** (long clips) or **loop-padding** (short clips).
- *Why uniform (not random) here:* val/test must be deterministic. Train-time temporal randomness lives in augmentation (Phase 6).
**Signer-invariance:** a first pass at **speed** invariance — fast and slow signers map to the same length.

### 3.7 Assemble tensor
Write `[C=3, T=64, V=75, M=1]` per clip → `processed/<dataset>/{train,val,test}.pkl`. This is the canonical input; **identical format for all six models.**

---

## Phase 4 — Skeleton graph construction

**What:** the adjacency matrix **A** over the 75 joints that every GCN convolves on.
**How:** three linked subgraphs —
1. **Body:** shoulder→elbow→wrist chains, shoulders↔neck, head points to shoulders.
2. **Left hand (21)** and **Right hand (21):** MediaPipe hand topology — wrist→each finger MCP→PIP→DIP→TIP.
3. **Cross-links:** pose wrist (15/16) ↔ the corresponding hand wrist, so arm motion and handshape share information.
**Why:** the graph must match the 75-joint layout exactly; reusing each paper's native NTU graph would silently break the comparison.
**Bone list:** define parent→child pairs here; the bone stream (Phase 5) is built from them.

> For **HA-GCN** this graph is first-class: its *hand-aware* layer expects explicit hand sub-graphs, which map directly onto our two 21-node hand blocks.

---

## Phase 5 — Multi-stream generation

All six reach their best accuracy as a **4-stream ensemble**. From the joint tensor derive:

| Stream | Definition | Why it helps / invariance note |
|---|---|---|
| **Joint** | normalized coords (Phase 3) | the pose configuration |
| **Bone** | vector between connected joints | **translation-invariant by construction** → extra signer/position robustness |
| **Joint motion** | joint(t+1) − joint(t) | captures dynamics; removes static-posture bias |
| **Bone motion** | bone(t+1) − bone(t) | dynamics of limb/finger structure |

Train each model **once per stream** → 4 checkpoints/model; fuse in Phase 10.

---

## Phase 6 — Data augmentation (train split only)

On-the-fly; **identical policy for all six**. Ranges are chosen to **simulate an unseen signer/camera without changing the sign's identity**:

| Augmentation | Range | Why this range (and the signer-invariance intent) |
|---|---|---|
| Rotation | **±13°** | mimics small camera/body tilt; larger would flip orientation-dependent signs |
| Scaling | **×[0.9, 1.1]** | residual body-size / distance variation after normalization |
| Translation | **±5% frame** | residual position jitter |
| Shear | **±0.1** | mild perspective differences |
| Speed (temporal) | **×[0.8, 1.2]** | **directly trains tempo/speed invariance across signers** |
| Random frame drop | **≤10% frames** | robustness to dropped detections |
| Joint masking | drop **1–3 joints** randomly | robustness to real occlusions, which are signer-specific |
| Coordinate noise | Gaussian **σ≈0.01–0.02** | sensor/extractor jitter |
| **Horizontal flip** | **OFF by default** | flipping swaps handedness and can **change a sign's meaning**; only enable per-dataset if labels are verified symmetric |

---

## Phase 7 — Splits, imbalance, labels

- **Signer-independent** train/val/test, saved as fixed index lists. Where the signer count allows, use **leave-one-signer-out (LOSO) cross-validation** — the gold standard for signer invariance.
- **Imbalance:** class-balanced sampling or class-weighted loss; always report **mean per-class accuracy** (Phase 11), not just Top-1.
- **Few-shot datasets:** CISLR (~1.5 samples/class) cannot be trained conventionally → use for one-shot evaluation only, or restrict to classes with enough samples, and state which.
- **Label map** consistent across splits, streams, and models.

> End of the pre-GCN pipeline. The tensor is now fixed and signer-normalized; only the model changes from here.

---

## Phase 8 — Model configuration (standardized)

Same `in_channels=3, T=64, V=75, M=1, graph=ours, num_classes=<dataset>` for all six; differ **only** in architecture.

| Model | What's distinctive | Config note for a fair run |
|---|---|---|
| **ST-GCN** | baseline spatial-temporal GCN, fixed graph | the reference everything is measured against |
| **MS-G3D** | multi-scale disentangled spatio-temporal (G3D) | heavier; keep its multi-scale windows |
| **CTR-GCN** | channel-wise topology refinement | lightweight, strong baseline |
| **InfoGCN** | information-bottleneck objective + self-attention topology | **keep its native loss term** — it's part of the model |
| **HD-GCN** | hierarchically decomposed graph + attention aggregation | keep its HD-graph module |
| **HA-GCN** | **hand-aware** conv (global body + 2 hand sub-graphs), **adaptive DropGraph**, multi-stream; built for SLR, benchmarked on AUTSL+INCLUDE | keep its hand sub-graphs + DropGraph; our 75-joint layout feeds it natively |

*Fairness rule:* each model keeps its own **intrinsic** components (InfoGCN's loss, HD-GCN's graph, HA-GCN's DropGraph) but shares **everything external** (data, graph layout, streams, budget).

---

## Phase 9 — Training setup

| Setting | Value | Why |
|---|---|---|
| Loss | cross-entropy + **label smoothing 0.1** | smoothing curbs over-confidence on a specific signer's style |
| Optimizer | **SGD + Nesterov, momentum 0.9** | the setting all these papers report best under; AdamW as fallback |
| Weight decay | **5e-4** | standard regularization for these GCNs |
| LR schedule | base **0.1** (SGD, batch 64) → **5-epoch warmup → cosine decay** | warmup stabilises the graph modules; cosine gives a clean final convergence |
| Epochs / batch | **~100 / 64** | enough to converge these sizes; same budget for all six |
| Precision / clipping | mixed precision, grad-clip | speed + stability |
| Runs | **4 streams × 6 models × 3–5 seeds** per dataset | streams for accuracy, seeds for honest variance |

Checkpoint on **best validation** accuracy; log curves.

---

## Phase 10 — Getting the *best* model (fairly)

1. **Hyperparameter tuning on val only**, with an **equal search budget per model** (same grid for LR, weight decay, dropout, T). Equal budget = the ranking reflects the architecture, not uneven tuning effort.
2. **4-stream late fusion:** weighted softmax sum, start **[joint 1.0, bone 1.0, joint-motion 0.5, bone-motion 0.5]**, tune weights on val. *Why these weights:* joint & bone encode the static configuration (primary signal); motion streams are complementary but noisier, so half weight. **This fusion is how these nets hit their published numbers — a single stream undersells all six.**
3. **Transfer learning (optional, powerful):** pretrain on a large skeleton set (NTU RGB+D or WLASL) → fine-tune per SL; apply to **all six or none**.
4. **Best-val checkpoint** → the test set is touched **once**.

---

## Phase 11 — Evaluation & comparison

Run every model through the **same** protocol.

| Metric | What it measures | Why we report it |
|---|---|---|
| **Top-1 accuracy** | exact recognition rate | the headline number |
| **Top-5 accuracy** | correct class in top 5 | tolerance for visually confusable signs; standard on large-vocab sets (WLASL) |
| **Mean per-class (macro) accuracy** | average recall over classes | datasets are imbalanced; macro weights rare signs equally |
| **Signer-independent accuracy** | accuracy on **unseen** signers | the project's primary objective |
| **Signer gap** = signer-dependent − signer-independent acc | drop when moving to unseen signers | **direct measure of signer invariance** — lower is better; a real comparison axis between the six |
| Confusion matrix / per-class recall | *which* signs fail | error analysis |
| **#Params, FLOPs, latency (ms/clip), train time** | model cost | a fair comparison weighs accuracy *against* compute/deployability |
| **Mean ± std over seeds + significance test** (Wilcoxon / paired t) | run-to-run stability | GCNs vary across seeds; single runs mislead — never claim a win without it |

Report per-dataset and per-SL; optional **cross-lingual** transfer (train one SL, test another).

---

## Phase 12 — Analysis & reporting

- **Leaderboard** per dataset: model × {Top-1, Top-5, macro-acc, **signer-gap**, params, FLOPs, latency}.
- **Accuracy-vs-cost** and **accuracy-vs-signer-gap** scatter plots (the practical "which model wins" views).
- **Error analysis:** confused sign pairs, per-SL difficulty, effect of hand occlusion/missing hands.
- **Ablations:** single- vs 4-stream; with/without rotation norm; with/without transfer learning; 75 vs 65 joints.

---

## Signer-invariance strategy (consolidated)

Ranked by impact, this is *why* the pipeline is built the way it is:

1. **Pose instead of RGB** — removes appearance, clothing, skin, lighting, background (Phase 2).
2. **Mid-shoulder centering** — removes signer position & camera pan (3.3).
3. **Shoulder-width scaling (per-clip constant)** — removes body size & camera distance (3.4).
4. **Shoulder-line rotation** — removes camera/body tilt (3.5).
5. **Temporal resample + speed augmentation** — removes signing-tempo differences (3.6, 6).
6. **Bone & motion streams** — translation-invariant / static-bias-free representations (5).
7. **Drop z** — removes an unreliable, camera/signer-dependent axis (0, 2).
8. **Signer-independent splits / LOSO + signer-gap metric** — *measures* invariance so you can prove it (7, 11).
9. **(Optional, advanced) domain-adversarial training** — a gradient-reversal signer-classifier head that actively strips signer identity from the learned features; apply identically to all six if used.

---

## Parameter choices — quick reference

| Parameter | Value | One-line reason |
|---|---|---|
| Joints V | 75 (33 pose + 42 hands) | hands = lexical signal, body = trajectory; face dropped |
| Channels C | (x, y, conf), z dropped | z unreliable → would add signer/camera noise |
| Clip length T | 64 frames | covers a full isolated sign; bounds compute |
| Confidence threshold | 0.5 | below it the landmark is a guess |
| Interpolation gap | ≤ 7 frames (~0.25 s) | recover short occlusions, don't hallucinate long ones |
| Center joint | mid-shoulder (11,12) | always visible, central, stable, crop-proof |
| Scale reference | shoulder width, per-clip median | stable anthropometric constant; preserves motion |
| Rotation norm | shoulders → horizontal | removes camera/body tilt |
| Aug rotation / scale / speed | ±13° / ×[0.9,1.1] / ×[0.8,1.2] | realistic unseen-signer variation, sign identity preserved |
| Horizontal flip | OFF | can change sign meaning |
| Fusion weights | 1 / 1 / 0.5 / 0.5 | joint+bone primary, motion complementary |
| Loss / smoothing | CE / 0.1 | curbs signer-style over-confidence |
| Optimizer / LR / WD | SGD-Nesterov / 0.1 cosine+warmup / 5e-4 | the regime these GCNs train best under |
| Epochs / batch | ~100 / 64 | converges; same budget for all six |
| Seeds | 3–5 | honest mean ± std + significance |

---

## Suggested repo structure

```
ISL-Project/
├── ISLR/                      # download scripts (done)
├── pose_extraction/           # Phase 2  (MediaPipe/RTMPose → raw .npy)
├── preprocess/                # Phases 3–7 → processed/*.pkl  (the signer-invariance core)
├── graphs/                    # Phase 4  (75-joint adjacency + bones, incl. hand sub-graphs)
├── configs/                   # Phase 8  (per-model × per-dataset)
├── models/                    # ST-GCN, MS-G3D, CTR-GCN, InfoGCN, HD-GCN, HA-GCN
├── train.py / sweep.py        # Phases 9–10
├── eval.py                    # Phase 11 (incl. signer-gap)
└── results/                   # Phase 12 leaderboards, plots, logs
```

**References grounding the key choices:** HA-GCN (Song et al., *Hand-aware graph convolution network for skeleton-based sign language recognition*, 2024); the 75-keypoint MediaPipe-Holistic-minus-face layout and neck-centred normalization are standard in recent SLR pose pipelines; MediaPipe documents its z-coordinate as unreliable.
