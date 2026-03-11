#!/usr/bin/env python3
"""
Set repository topics via GitHub API.
Run after the repo exists. Requires GITHUB_TOKEN with repo scope.
"""

import os
import sys

REPO = "AndriiShramko/4dgs-human-avatar-comfyui-nodes-dataset"
TOPICS = [
    "4d-gaussian-splatting",
    "comfyui",
    "comfyui-nodes",
    "human-avatar",
    "dataset",
    "neural-rendering",
    "4dgs",
    "3d-reconstruction",
]


def main():
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Set GITHUB_TOKEN environment variable (with repo scope)")
        sys.exit(1)

    try:
        import urllib.request
        import json
    except ImportError:
        print("Use Python 3 with urllib")
        sys.exit(1)

    url = f"https://api.github.com/repos/{REPO}/topics"
    data = json.dumps({"names": TOPICS}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.mercy-preview+json",
            "Content-Type": "application/json",
        },
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            out = json.loads(resp.read().decode())
            print("Topics set:", out.get("names", []))
    except Exception as e:
        print("Error:", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
