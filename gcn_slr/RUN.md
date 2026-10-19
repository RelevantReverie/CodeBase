# RUN — running the gcn_slr pipeline on Windows

Step-by-step to set up and run the whole pipeline on a fresh **Windows** machine.
All commands are **PowerShell**. Nothing in the code uses machine-specific paths.

---

## 0. What to copy over

Copy the `gcn_slr/` folder (and `ISLR/` if you'll download datasets there too).

```
gcn_slr\            # the pipeline (this folder)
ISLR\               # optional: dataset download scripts
```

---

## 1. Prerequisites

- **Python 3.9–3.12** (MediaPipe has no 3.13+ wheel yet). Get it from python.org and tick
  "Add Python to PATH" during install.
- A **GPU + CUDA** for training (CPU works for the smoke test and tiny runs, but full
  training on real data needs a GPU).

Check:
```powershell
python --version
```

---

## 2. Create an environment & install

```powershell
cd gcn_slr

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
```
> If activation is blocked by execution policy, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Install PyTorch** — pick the build for your machine:

```powershell
# GPU (CUDA 12.1):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Install the rest:**
```powershell
pip install -r requirements.txt
```
> `mediapipe` / `opencv-python` are only needed for Phase 2 (pose extraction).
> If your data already ships pose, you can skip them.

Verify CUDA is seen:
```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available())"
```

---

## 3. Verify the install (no data needed)

```powershell
python smoke_test.py
```
Expected: it prints each phase, lists all six models with `ok`, and ends with
`SMOKE TEST PASSED`. If this passes, the code is wired correctly on this machine.

---

## 4. Get the data

Use the download scripts in the sibling `ISLR\` folder — see `ISLR\README.md`:
```powershell
python ..\ISLR\dataset_download\download_isl.py --out ..\data
```
Several datasets are gated/registration-only; those scripts drop a
`DOWNLOAD_INSTRUCTIONS.txt` into each folder. You end up with raw videos (or, for
pose-shipping datasets, keypoints) per dataset.

---

## 5. Phase 2 — pose extraction (RGB video → keypoints)

Skip this if your dataset already provides pose (just produce the same
`[T,75,3]` layout yourself).

```powershell
python phase2_pose_extraction.py `
    --videos ..\data\ISL\INCLUDE\videos `
    --out    ..\data\ISL\INCLUDE\raw
```
Output: one `<clip_id>.npy` of shape `[T, 75, 3]` (x, y, confidence) per video.

---

## 6. Phase 3 — preprocess → training tensors

You need a **manifest CSV** describing each clip. Columns (header required):

```csv
clip_id,label,signer,split
clip0001,0,signerA,train
clip0002,0,signerB,val
clip0003,1,signerC,test
```
- `clip_id` matches `<clip_id>.npy` in the raw folder.
- `label` is an integer class id (0-based).
- `signer` is the signer id — **used to keep splits signer-independent**.
- `split` is `train`, `val`, or `test`.

Run:
```powershell
python phase3_preprocess.py `
    --raw-dir  ..\data\ISL\INCLUDE\raw `
    --manifest ..\data\ISL\INCLUDE\manifest.csv `
    --out-dir  processed\ISL
```
Output: `processed\ISL\train.pkl`, `val.pkl`, `test.pkl` — each a dict with
`data [N,3,64,75,1]`, `label [N]`, `signer`.

---

## 7. Phase 9 — train every model × every stream

One run = one model + one stream. Loop over all 6 models × 4 streams:

```powershell
foreach ($m in "st_gcn","ms_g3d","ctr_gcn","info_gcn","hd_gcn","ha_gcn") {
  foreach ($s in "joint","bone","joint_motion","bone_motion") {
    python phase9_train.py --model $m --stream $s `
      --config configs\$m.yaml `
      --train processed\ISL\train.pkl --val processed\ISL\val.pkl --out runs\ISL
  }
}
```

Useful flags: `--epochs 5` (quick test), `--num-class 263` (force class count).
Checkpoints land in `runs\ISL\<model>_<stream>_best.pt`.

---

## 8. Phase 11 — fuse streams & score (signer-independent test)

```powershell
python phase11_eval.py `
    --model ha_gcn `
    --ckpt-dir runs\ISL `
    --test processed\ISL\test.pkl `
    --streams joint bone joint_motion bone_motion
```
Prints Top-1, Top-5, mean per-class accuracy for the 4-stream fused model.
Repeat per model to build the leaderboard. For the **signer-gap** metric, also
build a *signer-dependent* split and compare (see `common.signer_gap`).

---

## 9. Repeat per sign language

Phases 6–8 are the same; just change the dataset path:
`processed\ASL`, `processed\BSL`, `processed\Auslan`, `processed\Turkish`.

---

## Resource notes

- **GPU memory:** `base_channels: 64`, `batch_size: 32` fits comfortably in ~6–8 GB.
  Lower `batch_size` in `configs\*.yaml` if you hit OOM.
- **Time:** ~100 epochs × 6 models × 4 streams is a lot of runs — start with
  `--epochs 5` to validate the whole flow on your real data before the full sweep.
- **Seeds:** set `seed` in the config; run 3–5 seeds for mean ± std (pipeline doc, Phase 11).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `python` not found | Reinstall Python with "Add to PATH", or use `py` instead of `python`. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then re-activate. |
| `mediapipe` won't install | Use Python 3.9–3.12; it has no 3.13 wheel yet. |
| `torch.cuda.is_available()` is False | You installed the CPU wheel — reinstall the CUDA build (§2). |
| Kaggle download fails | Put your token at `C:\Users\<you>\.kaggle\kaggle.json` (see `ISLR\README.md`). |
| Paths with spaces | Quote them: `"..\data\My Folder\..."`. |
| DataLoader hangs / slow | Keep `num_workers=0` (the default in `phase7_dataset.make_loader`). |
| `KeyError: num_class` or wrong class count | Pass `--num-class N`, or it's inferred from the train split's max label. |
| Out of memory | Lower `batch_size` and/or `base_channels` in `configs\*.yaml`. |

---

## One-shot sanity run on real data

Before the full sweep, confirm the real data trains end-to-end:
```powershell
python phase9_train.py --model ctr_gcn --stream joint --epochs 2 `
  --train processed\ISL\train.pkl --val processed\ISL\val.pkl --out runs\_sanity
python phase11_eval.py --model ctr_gcn --ckpt-dir runs\_sanity `
  --test processed\ISL\test.pkl --streams joint
```
If both complete and print metrics, you're ready for the full sweep (§7–8).
