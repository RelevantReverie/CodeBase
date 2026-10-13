#!/usr/bin/env python3
"""
Download the Indian Sign Language (ISL) ISLR datasets: INCLUDE, CISLR, FDMSE-ISL.
Each goes into its own folder under  <data-root>/ISL/ .

    python download_isl.py                 # download all
    python download_isl.py --list          # see the datasets
    python download_isl.py --only INCLUDE  # just one
    python download_isl.py --out /mnt/data # custom output location
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

DATASETS = [
    {
        "name": "INCLUDE",
        "folder": "INCLUDE",
        "method": "zenodo",
        "record": "4010759",  # VERIFY at https://github.com/AI4Bharat/INCLUDE if files are missing
        "homepage": "https://zenodo.org/record/4010759",
        "size": "18 GB",
        "instructions": (
            "INCLUDE / INCLUDE-50 (AI4Bharat, ACM MM 2020).\n"
            "This script pulls Zenodo record 4010759. If that record id is wrong or files\n"
            "are missing, find the current link in the official repo\n"
            "  https://github.com/AI4Bharat/INCLUDE\n"
            "and either update 'record' above or drop the archives into this folder manually."
        ),
    },
    {
        "name": "CISLR",
        "folder": "CISLR",
        "method": "manual",
        "homepage": "https://github.com/Exploration-Lab/CISLR",
        "size": "15 GB",
        "instructions": (
            "CISLR - Corpus for Indian Sign Language Recognition (EMNLP 2022).\n"
            "Get the video + pose data via the official repo (follow its README / data form):\n"
            "  https://github.com/Exploration-Lab/CISLR\n"
            "Place the extracted files in this folder."
        ),
    },
    {
        "name": "FDMSE-ISL",
        "folder": "FDMSE-ISL",
        "method": "manual",
        "homepage": "https://arxiv.org/abs/2407.14224",
        "size": "120 GB",
        "instructions": (
            "FDMSE-ISL (introduced with HWGAT, 2024; 2002 words x 20 videos = ~40,033 clips).\n"
            "Obtain from the authors' release linked in the paper / its project page:\n"
            "  https://arxiv.org/abs/2407.14224\n"
            "Place the files in this folder."
        ),
    },
]

if __name__ == "__main__":
    common.main("ISL", DATASETS)
