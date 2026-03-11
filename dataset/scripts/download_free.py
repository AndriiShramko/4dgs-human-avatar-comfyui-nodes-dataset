#!/usr/bin/env python3
"""
Download the free 4DGS human scan dataset from Hugging Face.
License: CC BY-NC-4.0. Non-commercial use only.
"""

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Download free 4DGS human scan dataset from Hugging Face")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/4dgs_free",
        help="Directory to save downloaded data",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="",
        help="Hugging Face dataset repo ID (e.g. org/4dgs-human-scans-free). Set when dataset is published.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Repo revision or branch",
    )
    args = parser.parse_args()

    if not args.repo_id:
        print(
            "No Hugging Face repo_id set. When the free dataset is published,\n"
            "run with: python download_free.py --repo_id YOUR_ORG/4dgs-human-scans-free --output_dir ./data/4dgs_free"
        )
        print("\nCreating output directory for future use:", os.path.abspath(args.output_dir))
        os.makedirs(args.output_dir, exist_ok=True)
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install huggingface_hub: pip install huggingface_hub")
        raise

    os.makedirs(args.output_dir, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
    )
    print("Downloaded to:", path)


if __name__ == "__main__":
    main()
