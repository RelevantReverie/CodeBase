#!/usr/bin/env python3
"""
Shared helpers for the ISLR dataset-download scripts.

Portable by design:
  * Pure Python 3.8+ standard library for the core logic.
  * A few OPTIONAL third-party tools are used only when a dataset needs them
    (kaggle, yt-dlp, gdown, huggingface_hub). If one is missing, the script
    writes clear instructions into that dataset's folder instead of crashing.
  * No absolute paths from the machine that generated this. Output location is
    derived from this file's location, or set with  --out .

A per-language script just does:

    import common
    common.main("ASL", DATASETS)

Output layout (created automatically):

    <data-root>/<Language>/<Dataset>/...

Default <data-root> is  ISLR/data  (a sibling of this dataset_download/ folder).
Override with:  python download_asl.py --out /mnt/bigdisk/islr_data
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def have(tool: str) -> bool:
    """True if an executable is on PATH."""
    return shutil.which(tool) is not None


def run(cmd) -> None:
    log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def default_data_root() -> Path:
    # common.py lives in  ISLR/dataset_download/ ; data root is  ISLR/data
    return Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# generic download / extract
# --------------------------------------------------------------------------- #
def _progress(done: int, total: int) -> None:
    if total > 0:
        pct = done * 100 // total
        bar = "#" * (pct // 4)
        sys.stdout.write(f"\r    {pct:3d}%  |{bar:<25}| {done >> 20} / {total >> 20} MiB")
    else:
        sys.stdout.write(f"\r    {done >> 20} MiB")
    sys.stdout.flush()


def download_file(url: str, dest_path: Path, chunk: int = 1 << 20) -> Path:
    """Stream a URL to disk with a progress bar. Resumes nothing, but skips a
    file that already exists and is non-empty."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        log(f"exists, skip: {dest_path.name}")
        return dest_path
    tmp = dest_path.with_name(dest_path.name + ".part")
    log(f"GET {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ISLR-downloader/1.0"})
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            block = r.read(chunk)
            if not block:
                break
            f.write(block)
            done += len(block)
            _progress(done, total)
    print()
    tmp.rename(dest_path)
    return dest_path


def extract(archive: Path, dest_dir: Path) -> None:
    archive, dest_dir = Path(archive), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        log(f"unzip {archive.name}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    elif tarfile.is_tarfile(archive):
        log(f"untar {archive.name}")
        with tarfile.open(archive) as t:
            try:
                t.extractall(dest_dir, filter="data")  # py3.12+ safe extraction
            except TypeError:
                t.extractall(dest_dir)
    else:
        log(f"not an archive, left as-is: {archive.name}")


# --------------------------------------------------------------------------- #
# provider-specific helpers
# --------------------------------------------------------------------------- #
def manual_note(folder: Path, instructions: str) -> None:
    """Record how to obtain a dataset that cannot be fetched automatically."""
    write_text(Path(folder) / "DOWNLOAD_INSTRUCTIONS.txt", instructions.strip() + "\n")
    log("  -> needs manual access; wrote DOWNLOAD_INSTRUCTIONS.txt")


def zenodo_record(record_id: str, folder: Path) -> None:
    api = f"https://zenodo.org/api/records/{record_id}"
    log(f"Zenodo record {record_id}")
    req = urllib.request.Request(api, headers={"User-Agent": "ISLR-downloader/1.0"})
    with urllib.request.urlopen(req) as r:
        data = json.load(r)
    files = data.get("files", [])
    if not files:
        raise RuntimeError("Zenodo record returned no files")
    for f in files:
        name = f.get("key") or f["links"]["self"].split("/")[-1]
        url = f["links"]["self"]
        out = download_file(url, Path(folder) / name)
        if name.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
            extract(out, folder)


def kaggle_dataset(slug: str, folder: Path, instructions: str) -> None:
    if not have("kaggle"):
        manual_note(folder,
                    "Kaggle CLI not found.\n"
                    "  pip install kaggle   (then put your token at ~/.kaggle/kaggle.json)\n"
                    f"  kaggle datasets download -d {slug} -p . --unzip\n\n" + instructions)
        return
    run(["kaggle", "datasets", "download", "-d", slug, "-p", str(folder), "--unzip"])


def gdrive(file_or_folder_id: str, folder: Path, instructions: str) -> None:
    if not have("gdown"):
        manual_note(folder,
                    "gdown not found.\n  pip install gdown\n"
                    f"  gdown --fuzzy {file_or_folder_id} -O <file>   (or --folder for a folder id)\n\n"
                    + instructions)
        return
    # works for a bare file id, a share URL, or a folder id/URL
    if "drive.google.com" in file_or_folder_id and "folders" in file_or_folder_id:
        run(["gdown", "--folder", file_or_folder_id, "-O", str(folder)])
    else:
        run(["gdown", "--fuzzy", file_or_folder_id, "-O", str(folder) + "/"])


def hf_dataset(repo_id: str, folder: Path, instructions: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception:
        manual_note(folder,
                    "huggingface_hub not found.\n  pip install huggingface_hub\n"
                    f"  huggingface-cli download {repo_id} --repo-type dataset --local-dir .\n\n"
                    + instructions)
        return
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(folder))


def msasl(ds: dict, folder: Path) -> None:
    """MS-ASL ships annotations (JSON) + YouTube URLs; videos are fetched with
    yt-dlp. Step 1 gets the metadata zip (if a direct url is configured),
    step 2 downloads the source videos listed in the JSON."""
    meta_url = ds.get("meta_url")
    if meta_url:
        z = download_file(meta_url, Path(folder) / "MS-ASL.zip")
        extract(z, folder)

    jsons = list(Path(folder).rglob("*.json"))
    if not jsons:
        manual_note(folder, ds["instructions"])
        return
    if not have("yt-dlp"):
        manual_note(folder, "Metadata present but yt-dlp is missing.\n  pip install yt-dlp\n\n"
                            + ds["instructions"])
        return

    urls = set()
    for j in jsons:
        try:
            items = json.load(open(j, encoding="utf-8"))
        except Exception:
            continue
        if isinstance(items, dict):
            items = items.get("data", []) or []
        for it in items:
            if isinstance(it, dict):
                u = it.get("url") or it.get("video_url")
                if u:
                    urls.add(u)
    vids = Path(folder) / "videos"
    vids.mkdir(exist_ok=True)
    log(f"MS-ASL: {len(urls)} unique source videos via yt-dlp")
    for u in sorted(urls):
        try:
            run(["yt-dlp", "-f", "mp4/best", "-o", str(vids / "%(id)s.%(ext)s"), u])
        except subprocess.CalledProcessError as e:
            log(f"  skip (unavailable): {u}  ({e})")
    write_text(Path(folder) / "NOTE_clips.txt",
               "Full source videos are in ./videos .\n"
               "Cut per-sign clips with the start_time/end_time fields in the JSON, e.g.:\n"
               "  ffmpeg -ss <start> -to <end> -i videos/<id>.mp4 clips/<label>_<n>.mp4\n")


# --------------------------------------------------------------------------- #
# per-dataset dispatch
# --------------------------------------------------------------------------- #
def write_info(folder: Path, ds: dict) -> None:
    info = {
        "name": ds.get("name"),
        "method": ds.get("method"),
        "homepage": ds.get("homepage"),
        "approx_size": ds.get("size"),
        "notes": ds.get("instructions", ""),
    }
    write_text(Path(folder) / "SOURCE.json", json.dumps(info, indent=2, ensure_ascii=False))


def process(ds: dict, lang_root: Path) -> None:
    folder = lang_root / ds["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    method = ds["method"]
    log(f"=== {ds['name']}  [{method}]  (~{ds.get('size', '?')})  ->  {folder}")
    try:
        if method == "http":
            fname = ds.get("filename") or ds["url"].split("/")[-1]
            out = download_file(ds["url"], folder / fname)
            if ds.get("extract"):
                extract(out, folder)
        elif method == "zenodo":
            zenodo_record(ds["record"], folder)
        elif method == "kaggle":
            kaggle_dataset(ds["slug"], folder, ds.get("instructions", ""))
        elif method == "gdrive":
            gdrive(ds["gdrive"], folder, ds.get("instructions", ""))
        elif method == "hf":
            hf_dataset(ds["repo"], folder, ds.get("instructions", ""))
        elif method == "msasl":
            msasl(ds, folder)
        elif method == "manual":
            manual_note(folder, ds["instructions"])
        else:
            log(f"  unknown method: {method}")
    except Exception as e:  # keep going with the other datasets
        log(f"  !! {ds['name']} failed: {e}")
        manual_note(folder, ds.get("instructions",
                                   f"Automated download failed: {e}\nSource: {ds.get('homepage', '')}"))
    finally:
        write_info(folder, ds)


# --------------------------------------------------------------------------- #
# entry point used by every per-language script
# --------------------------------------------------------------------------- #
def main(language: str, datasets: list) -> None:
    ap = argparse.ArgumentParser(
        description=f"Download the {language} ISLR datasets, each into its own folder.")
    ap.add_argument("--out", default=None,
                    help="data root (default: ISLR/data next to this script)")
    ap.add_argument("--only", nargs="*", metavar="NAME",
                    help="only these datasets (match on name or folder)")
    ap.add_argument("--list", action="store_true", help="list the datasets and exit")
    args = ap.parse_args()

    if args.list:
        print(f"\n{language} ISLR datasets:")
        for d in datasets:
            print(f"  {d['name']:<18} {d['method']:<8} ~{d.get('size', '?'):<16} {d.get('homepage', '')}")
        return

    data_root = Path(args.out).expanduser().resolve() if args.out else default_data_root()
    lang_root = data_root / language
    lang_root.mkdir(parents=True, exist_ok=True)

    selection = datasets
    if args.only:
        want = {s.lower() for s in args.only}
        selection = [d for d in datasets
                     if d["name"].lower() in want or d["folder"].lower() in want]
        if not selection:
            log(f"no dataset matched {args.only}; options: "
                + ", ".join(d['name'] for d in datasets))
            return

    log(f"Language: {language}   Output root: {lang_root}")
    for d in selection:
        process(d, lang_root)
    log("Done. Open DOWNLOAD_INSTRUCTIONS.txt inside any folder that still needs manual access.")
