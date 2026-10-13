#!/usr/bin/env python3
"""
Download the Turkish Sign Language (TID) ISLR datasets: BosphorusSign22k, AUTSL.
Each goes into its own folder under  <data-root>/Turkish/ .

Both require a short registration / request, so this script creates the folders
and writes exact access instructions into each one.

    python download_turkish.py
    python download_turkish.py --list
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

DATASETS = [
    {
        "name": "BosphorusSign22k",
        "folder": "BosphorusSign22k",
        "method": "manual",
        "homepage": "https://www.bosphorussign.com",
        "size": "100 GB (RGB+depth+skeleton)",
        "instructions": (
            "BosphorusSign22k (2020) - 744 classes, RGB-D + skeleton + OpenPose.\n"
            "Request access via the dataset site:\n"
            "  https://www.bosphorussign.com\n"
            "After approval, download into this folder. (Pose-only is ~3 GB.)"
        ),
    },
    {
        "name": "AUTSL",
        "folder": "AUTSL",
        "method": "manual",
        "homepage": "https://cvml.ankara.edu.tr/datasets/",
        "size": "60 GB (RGB+depth+skeleton)",
        "instructions": (
            "AUTSL (2020) - 226 signs, 38K videos, RGB-D + skeleton.\n"
            "Register at:\n"
            "  https://cvml.ankara.edu.tr/datasets/\n"
            "to receive the Google-Drive download links, then place the files here.\n"
            "If you get a Drive id, switch this entry to method='gdrive' and set 'gdrive'."
        ),
    },
]

if __name__ == "__main__":
    common.main("Turkish", DATASETS)
