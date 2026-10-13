#!/usr/bin/env python3
"""
Download the British Sign Language (BSL) ISLR datasets: BSLDict, BSL-1K.
Each goes into its own folder under  <data-root>/BSL/ .

Both are access-controlled (data-licence / BBC agreement), so this script
creates the folders and writes exact access instructions into each one.

    python download_bsl.py
    python download_bsl.py --list
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

DATASETS = [
    {
        "name": "BSLDict",
        "folder": "BSLDict",
        "method": "manual",
        "homepage": "https://www.robots.ox.ac.uk/~vgg/data/bsldict/",
        "size": "5 GB",
        "instructions": (
            "BSLDict - visual sign dictionary (Oxford VGG, 2020).\n"
            "Request / download (data-licence agreement) at:\n"
            "  https://www.robots.ox.ac.uk/~vgg/data/bsldict/\n"
            "Place the files in this folder."
        ),
    },
    {
        "name": "BSL-1K",
        "folder": "BSL-1K",
        "method": "manual",
        "homepage": "https://www.robots.ox.ac.uk/~vgg/research/bsl1k/",
        "size": "400 GB (gated)",
        "instructions": (
            "BSL-1K (ECCV 2020) is GATED - BBC licensing, non-commercial use only.\n"
            "Apply for access via the VGG project page:\n"
            "  https://www.robots.ox.ac.uk/~vgg/research/bsl1k/\n"
            "After approval, download the data into this folder."
        ),
    },
]

if __name__ == "__main__":
    common.main("BSL", DATASETS)
