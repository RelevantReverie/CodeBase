# ISLR dataset downloaders

Scripts to fetch the **isolated** sign-language recognition (ISLR) datasets used in
this project, one per sign language. Each dataset lands in its **own folder**.

```
ISLR/
├── README.md
├── requirements.txt
├── dataset_download/
│   ├── common.py              # shared helpers (download / extract / providers)
│   ├── download_isl.py        # INCLUDE, CISLR, FDMSE-ISL
│   ├── download_asl.py        # WLASL, ASL Citizen, MS-ASL
│   ├── download_bsl.py        # BSLDict, BSL-1K
│   ├── download_auslan.py     # MM-WLAuslan, Auslan (UCI)
│   └── download_turkish.py    # BosphorusSign22k, AUTSL
└── data/                      # created on first run
    ├── ISL/INCLUDE/ …
    ├── ASL/WLASL/ …
    └── …
```

## Portability

- **Pure Python 3.8+ standard library** for the core logic — runs on Windows, Linux, or macOS.
- **No machine-specific paths.** Output defaults to `ISLR/data` (next to the scripts); override with `--out`.
- Optional helpers (`kaggle`, `yt-dlp`, `gdown`, `huggingface_hub`) are used only when a
  dataset needs them. If one is missing, the script writes `DOWNLOAD_INSTRUCTIONS.txt`
  into that dataset's folder instead of failing.

To move to another computer, copy the whole `ISLR/` folder (or just `dataset_download/`).

## Usage

```bash
# from ISLR/dataset_download/
python download_isl.py                      # download all ISL datasets
python download_asl.py --list               # list without downloading
python download_asl.py --only WLASL         # just one dataset
python download_turkish.py --out /mnt/data  # send output elsewhere (e.g. a big disk)
```

Run each language's script independently. Re-running skips files already present.

## Setup for the automated ones

```bash
pip install -r requirements.txt   # or install individual tools as needed
```

- **Kaggle** (WLASL, ASL Citizen): create an API token at kaggle.com → Account →
  "Create New API Token" and save it to `~/.kaggle/kaggle.json`
  (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).
- **yt-dlp** (MS-ASL): needed to pull the source videos from YouTube.

## What is automated vs. manual

| Language | Dataset | How it downloads |
|---|---|---|
| ISL | INCLUDE | Zenodo API (verify record id if files are missing) |
| ISL | CISLR | manual — official repo / data form |
| ISL | FDMSE-ISL | manual — authors' release |
| ASL | WLASL | Kaggle mirror |
| ASL | ASL Citizen | Kaggle mirror (official MS link in notes) |
| ASL | MS-ASL | metadata + `yt-dlp` (videos from YouTube) |
| BSL | BSLDict | manual — VGG data licence |
| BSL | BSL-1K | manual — **gated** (BBC licence) |
| Auslan | MM-WLAuslan | manual — project page (huge; grab pose only) |
| Auslan | Auslan (UCI) | direct HTTP from UCI ML Repository |
| Turkish | BosphorusSign22k | manual — request form |
| Turkish | AUTSL | manual — registration → Google Drive |

Several of these datasets are access-controlled or distribute via registration links that
can't be hard-coded. For those, the script still creates the folder and drops in
`DOWNLOAD_INSTRUCTIONS.txt` with the exact page to use. Where a dataset later hands you a
Google-Drive link, switch that entry's `method` to `gdrive` (and set `gdrive`) — the plumbing
is already in `common.py`.

> These are skeleton-based GCN datasets. If you only need pose/keypoints, prefer the
> pose-shipping variants where offered (e.g. MM-WLAuslan pose, BosphorusSign22k skeleton)
> to avoid terabytes of raw video.
