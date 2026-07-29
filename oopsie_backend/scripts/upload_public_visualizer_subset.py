#!/usr/bin/env python3
"""Upload the scrubbed public visualizer subset to Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_SUBSET_ROOT = Path(
    "/home/arpit/test_projects/oopsiedata/oopsie-docs/docs/assets/visualizer-public"
)
DEFAULT_REPO_ID = "OopsieData-Submissions/Visualizer_Previews"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-root", type=Path, default=DEFAULT_SUBSET_ROOT)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF write token (defaults to $HF_TOKEN).",
    )
    args = parser.parse_args()

    if not args.subset_root.is_dir():
        raise SystemExit(f"Missing subset directory: {args.subset_root}")
    meta = args.subset_root / "metadata.json"
    if not meta.is_file():
        raise SystemExit(f"Missing {meta}; run export_public_visualizer_subset first.")

    api = HfApi(token=args.token)
    print(f"Uploading {args.subset_root} -> {args.repo_id}", flush=True)
    api.upload_folder(
        folder_path=str(args.subset_root),
        repo_id=args.repo_id,
        repo_type="dataset",
        ignore_patterns=["**/.DS_Store", "**/.*"],
    )
    print(
        "Done. Resolve base: "
        f"https://huggingface.co/datasets/{args.repo_id}/resolve/main/"
    )


if __name__ == "__main__":
    main()
