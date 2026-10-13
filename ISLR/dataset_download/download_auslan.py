#!/usr/bin/env python3
"""
Download the Australian Sign Language (Auslan) ISLR datasets: MM-WLAuslan,
Auslan (UCI / Kadous). Each goes into its own folder under  <data-root>/Auslan/ .

    python download_auslan.py
    python download_auslan.py --list
    python download_auslan.py --only "Auslan (UCI)"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

DATASETS = [
    {
        "name": "MM-WLAuslan",
        "folder": "MM-WLAuslan",
        "method": "manual",
        "homepage": "https://uq-cvlab.github.io/MM-WLAuslan-Dataset/",
        "size": "1-3 TB full / pose ~30 GB",
        "instructions": (
            "MM-WLAuslan (NeurIPS 2024) - multi-view, multi-modal, 282K+ videos.\n"
            "Download links are on the project page:\n"
            "  https://uq-cvlab.github.io/MM-WLAuslan-Dataset/\n"
            "It is very large; if you only need skeletons, grab the POSE modality only.\n"
            "If they provide a Google-Drive id, switch this entry to method='gdrive' and\n"
            "set 'gdrive' to that id to automate the download."
        ),
    },
    {
        "name": "Auslan (UCI)",
        "folder": "Auslan-UCI",
        "method": "http",
        # UCI ML Repository dataset 58: "Australian Sign Language signs (High Quality)".
        "url": "https://archive.ics.uci.edu/static/public/58/australian+sign+language+signs+high+quality.zip",
        "filename": "auslan-hq.zip",
        "extract": True,
        "homepage": "https://archive.ics.uci.edu/dataset/58/",
        "size": "<50 MB",
        "instructions": (
            "Auslan high-quality (Kadous) - UCI ML Repository, dataset 58.\n"
            "If the direct URL 404s, download the zip from the dataset page:\n"
            "  https://archive.ics.uci.edu/dataset/58/australian+sign+language+signs+high+quality\n"
            "and unzip it into this folder. (Sensor-glove data, not video - classic baseline.)"
        ),
    },
]

if __name__ == "__main__":
    common.main("Auslan", DATASETS)
