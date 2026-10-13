#!/usr/bin/env python3
"""
Download the American Sign Language (ASL) ISLR datasets: WLASL, ASL Citizen, MS-ASL.
Each goes into its own folder under  <data-root>/ASL/ .

    python download_asl.py
    python download_asl.py --list
    python download_asl.py --only WLASL "ASL Citizen"
    python download_asl.py --out /mnt/data

Needs the Kaggle CLI for WLASL + ASL Citizen (pip install kaggle, token at
~/.kaggle/kaggle.json) and yt-dlp for MS-ASL videos (pip install yt-dlp).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

DATASETS = [
    {
        "name": "WLASL",
        "folder": "WLASL",
        "method": "kaggle",
        "slug": "risangbaskoro/wlasl-processed",
        "homepage": "https://github.com/dxli94/WLASL",
        "size": "6 GB",
        "instructions": (
            "WLASL2000 (WACV 2020).\n"
            "This script uses the Kaggle mirror 'risangbaskoro/wlasl-processed'.\n"
            "For raw videos straight from source, use the official repo (downloads from\n"
            "YouTube with yt-dlp):  https://github.com/dxli94/WLASL"
        ),
    },
    {
        "name": "ASL Citizen",
        "folder": "ASL_Citizen",
        "method": "kaggle",
        "slug": "abd0kamel/asl-citizen",
        "homepage": "https://www.microsoft.com/en-us/research/project/asl-citizen/",
        "size": "43 GB",
        "instructions": (
            "ASL Citizen (Microsoft, NeurIPS 2023).\n"
            "This script uses the Kaggle mirror 'abd0kamel/asl-citizen'.\n"
            "Official download (ASL_Citizen.zip, 42.8 GB):\n"
            "  https://www.microsoft.com/en-us/download/details.aspx?id=105253"
        ),
    },
    {
        "name": "MS-ASL",
        "folder": "MS-ASL",
        "method": "msasl",
        # Set 'meta_url' to the direct MS-ASL.zip link to fully automate step 1.
        "meta_url": "",
        "homepage": "https://www.microsoft.com/en-us/download/details.aspx?id=100121",
        "size": "25 GB (videos via YouTube)",
        "instructions": (
            "MS-ASL (Microsoft, BMVC 2019) ships annotations + YouTube URLs.\n"
            "1) Download MS-ASL.zip (metadata) from\n"
            "     https://www.microsoft.com/en-us/download/details.aspx?id=100121\n"
            "   and unzip it into this folder (so the *.json files are here).\n"
            "2) Re-run this script: it fetches the listed videos with yt-dlp into ./videos .\n"
            "   (Set 'meta_url' in this script to the direct .zip link to skip step 1.)"
        ),
    },
]

if __name__ == "__main__":
    common.main("ASL", DATASETS)
