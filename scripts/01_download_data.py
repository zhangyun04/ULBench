#!/usr/bin/env python3
"""Download and extract VLM-UB datasets from Hugging Face.

Downloads zip/tar archives from ``Outloud4u/VLM-UB`` into ``data/``,
then extracts them so that paths match the config files.

Usage:
    # Download everything
    python scripts/01_download_data.py

    # Download only specific datasets
    python scripts/01_download_data.py --datasets coco,AID,LAD

    # Download only, skip extraction
    python scripts/01_download_data.py --no_extract

    # Dry run — show what would be downloaded
    python scripts/01_download_data.py --dry_run
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

HF_REPO_ID = "Outloud4u/VLM-UB"

# ── Dataset registry ─────────────────────────────────────────────────────
# Each entry:
#   key          = user-facing dataset name (used with --datasets)
#   hf_folder    = folder name in the HF repo
#   archives     = list of archive files inside that folder
#   extract_to   = local directory under data/ to extract into
#   post_extract = optional function(data_root, extract_to) for path fixups
#
# Final layout must match the `root:` paths in vqa_gen/configs/*.yaml:
#   AID          → data/AID/
#   coco         → data/coco/
#   LAD          → data/LAD/
#   LOGO-2K+     → data/LOGO-2K+/Logo-2K+/
#   celebrity    → data/celebrity_faces/Celebrity Faces Dataset/
#   mit_indoor67 → data/mit_indoor67/

def _fixup_logo2kplus(data_root, extract_to):
    """Logo-2K+ zip extracts to Logo-2K+/ inside extract_to."""
    inner = data_root / extract_to / "Logo-2K+"
    if not inner.exists():
        # Zip might extract directly; check for any subdirectory
        children = list((data_root / extract_to).iterdir())
        dirs = [c for c in children if c.is_dir()]
        if len(dirs) == 1 and dirs[0].name != "Logo-2K+":
            dirs[0].rename(inner)


def _fixup_celebrity(data_root, extract_to):
    """Celebrity zip extracts to 'Celebrity Faces Dataset/' inside extract_to."""
    expected = data_root / extract_to / "Celebrity Faces Dataset"
    if expected.exists():
        return
    # Check if it extracted directly into the folder
    children = list((data_root / extract_to).iterdir())
    dirs = [c for c in children if c.is_dir()]
    if len(dirs) == 1 and dirs[0].name != "Celebrity Faces Dataset":
        dirs[0].rename(expected)


def _fixup_aid(data_root, extract_to):
    """AID zip extracts as AID/<categories> inside extract_to → flatten."""
    nested = data_root / extract_to / "AID"
    target = data_root / extract_to
    if nested.exists() and nested.is_dir():
        # Move all contents up one level
        for item in list(nested.iterdir()):
            dest = target / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        # Remove the now-empty nested dir
        if nested.exists():
            shutil.rmtree(nested)


DATASETS = {
    "AID": {
        "hf_folder": "AID",
        "archives": ["AID_dataset.zip"],
        "extract_to": "AID",
        "post_extract": _fixup_aid,
    },
    "coco": {
        "hf_folder": "coco",
        "archives": [
            "annotations_trainval2017.zip",
            "train2017.zip",
            "val2017.zip",
        ],
        "extract_to": "coco",
    },
    "LAD": {
        "hf_folder": "LAD",
        "archives": ["LAD_images.zip", "LAD_annotations.zip"],
        "extract_to": "LAD",
    },
    "LOGO-2K+": {
        "hf_folder": "Logo-2K+",
        "archives": ["Logo-2K+.zip"],
        "extract_to": "LOGO-2K+",
        "post_extract": _fixup_logo2kplus,
    },
    "celebrity_faces": {
        "hf_folder": "celebrity-face",
        "archives": ["celebrity-face-image-dataset.zip"],
        "extract_to": "celebrity_faces",
        "post_extract": _fixup_celebrity,
    },
    "mit_indoor67": {
        "hf_folder": "mit_indoor67",
        "archives": ["indoorCVPR_09.tar"],
        "extract_to": "mit_indoor67",
    },
    # SpatialMQA: uses COCO images + loads from HF hub (liuziyan/SpatialMQA)
    # at VQA-gen time. No separate download needed.
}


# ── Download & extract ────────────────────────────────────────────────────

def _extract_archive(archive_path: Path, dest_dir: Path):
    """Extract a .zip or .tar/.tar.gz archive into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()

    if name.endswith(".zip"):
        print(f"    Extracting {archive_path.name} ...")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(dest_dir)
    elif name.endswith((".tar", ".tar.gz", ".tgz")):
        print(f"    Extracting {archive_path.name} ...")
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(dest_dir)
    else:
        print(f"    WARNING: Unknown archive format: {archive_path.name}, skipping.",
              file=sys.stderr)
        return

    print(f"    Extracted to {dest_dir}/")


def download_and_extract(data_root: Path, datasets: list[str] | None,
                         dry_run: bool, no_extract: bool,
                         local_hf_repo: Path | None = None):
    if local_hf_repo is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            print("ERROR: huggingface_hub not installed.  pip install huggingface_hub",
                  file=sys.stderr)
            sys.exit(1)

    data_root.mkdir(parents=True, exist_ok=True)

    # Filter
    if datasets:
        requested = set(datasets)
        selected = {k: v for k, v in DATASETS.items() if k in requested}
        missing = requested - set(DATASETS.keys())
        if missing:
            print(f"WARNING: Unknown dataset names (skipped): {missing}",
                  file=sys.stderr)
            print(f"  Available: {sorted(DATASETS.keys())}", file=sys.stderr)
    else:
        selected = DATASETS

    if not selected:
        print("Nothing to download.")
        return

    if local_hf_repo:
        print(f"HF repo    : {local_hf_repo.resolve()} (local)")
    else:
        print(f"HF repo    : {HF_REPO_ID}")
    print(f"Data root  : {data_root.resolve()}")
    print(f"Datasets   : {sorted(selected.keys())}")
    print()

    if dry_run:
        print("[DRY RUN] Would download and extract:")
        for name, info in sorted(selected.items()):
            hf = info["hf_folder"]
            local = info["extract_to"]
            for arc in info["archives"]:
                print(f"  {HF_REPO_ID}/{hf}/{arc} → data/{local}/")
        return

    for name, info in sorted(selected.items()):
        hf_folder = info["hf_folder"]
        extract_to = info["extract_to"]
        archives = info["archives"]
        post_extract = info.get("post_extract")

        if local_hf_repo:
            # Use already-cloned local HF repo — skip download entirely
            hf_local = local_hf_repo / hf_folder
            print(f"[{name}] Using local repo: {hf_local}/")
            if not hf_local.exists():
                print(f"  WARNING: Folder not found in local repo: {hf_local}",
                      file=sys.stderr)
                continue
        else:
            print(f"[{name}] Downloading from {HF_REPO_ID}/{hf_folder}/ ...")
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                local_dir=str(data_root / "_hf_download"),
                allow_patterns=[f"{hf_folder}/*"],
            )
            hf_local = data_root / "_hf_download" / hf_folder
            print(f"  Downloaded to {hf_local}/")

        if no_extract:
            print(f"  Skipping extraction (--no_extract).")
            continue

        # Extract each archive
        dest = data_root / extract_to
        for arc_name in archives:
            arc_path = hf_local / arc_name
            if not arc_path.exists():
                print(f"  WARNING: Archive not found: {arc_path}", file=sys.stderr)
                continue
            _extract_archive(arc_path, dest)

        # Post-extraction fixups (rename directories, etc.)
        if post_extract:
            post_extract(data_root, extract_to)

        print(f"  OK: {name} → {dest}/")
        print()

    # Clean up the HF download cache (only if we actually downloaded)
    hf_dl = data_root / "_hf_download"
    if hf_dl.exists() and not no_extract and local_hf_repo is None:
        print("Cleaning up download cache ...")
        shutil.rmtree(hf_dl)

    print("All downloads complete.")
    print()
    print("Expected data layout:")
    for name, info in sorted(selected.items()):
        print(f"  data/{info['extract_to']}/")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract VLM-UB datasets from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data_root", default="data",
        help="Local directory to download into (default: data/).",
    )
    parser.add_argument(
        "--datasets", default=None,
        help="Comma-separated dataset names to download. "
             "Default: all.  Available: " + ", ".join(sorted(DATASETS)),
    )
    parser.add_argument(
        "--no_extract", action="store_true",
        help="Download archives only, skip extraction.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Show what would be downloaded without actually doing it.",
    )
    parser.add_argument(
        "--local_hf_repo", default=None,
        help="Path to a locally cloned HF repo (e.g. VLM_UB/). "
             "Skips download and extracts directly from this folder.",
    )
    args = parser.parse_args()

    ds_list = [d.strip() for d in args.datasets.split(",")] if args.datasets else None
    local_repo = Path(args.local_hf_repo) if args.local_hf_repo else None
    download_and_extract(Path(args.data_root), ds_list, args.dry_run, args.no_extract,
                         local_hf_repo=local_repo)


if __name__ == "__main__":
    main()
