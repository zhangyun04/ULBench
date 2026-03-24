"""AID (Aerial Image Dataset) adapter for the outdoor scene-type VQA pipeline.

Loads the AID dataset from a local directory organized as
``<root>/CategoryName/image_NNNN.jpg`` and yields
:class:`CanonicalSample` objects compatible with the shared pipeline.

AID has no official train/test split files; the adapter iterates
all images from each category directory and the pipeline's
class-level splitting handles forget/retain/test assignment.

Reference: https://captain-whu.github.io/AID/
"""

import json
import logging
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample

logger = logging.getLogger(__name__)


class AIDAdapter:
    """Adapter for the Aerial Image Dataset (AID).

    Parameters
    ----------
    dataset_root : str
        Root directory of the AID dataset (contains category sub-dirs).
    category_metadata_path : str, optional
        Path to JSON file mapping directory names to
        ``{"display_name": ..., "supercategory": ...}``.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif"}

    def __init__(self, dataset_root, category_metadata_path=None):
        self.dataset_root = Path(dataset_root)

        self.wnid_to_class: dict[str, str] = {}
        self.wnid_to_superclass: dict[str, str] = {}

        if category_metadata_path:
            self._load_category_metadata(category_metadata_path)
        else:
            self._discover_categories()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_category_metadata(self, path):
        """Load category JSON with display names and supercategories."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        for dir_name, info in data.items():
            display = info.get("display_name", dir_name)
            supercat = info.get("supercategory", "unknown")
            self.wnid_to_class[dir_name] = display
            self.wnid_to_superclass[dir_name] = supercat

        logger.info(
            "Loaded %d AID categories from %s", len(self.wnid_to_class), path
        )

    def _discover_categories(self):
        """Fallback: discover categories from directory tree."""
        if not self.dataset_root.exists():
            logger.warning("Dataset root not found: %s", self.dataset_root)
            return

        for cat_dir in sorted(self.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            dir_name = cat_dir.name
            self.wnid_to_class[dir_name] = dir_name.lower()
            self.wnid_to_superclass[dir_name] = "unknown"

        logger.info(
            "Discovered %d AID categories from filesystem",
            len(self.wnid_to_class),
        )

    # ------------------------------------------------------------------
    # Sample iteration (round-robin across categories)
    # ------------------------------------------------------------------

    def _collect_per_category(self):
        """Return ``{dir_name: [image_relpath, ...]}``, sorted within each category."""
        per_cat: dict[str, list[str]] = {}
        if not self.dataset_root.exists():
            logger.warning("Dataset root not found: %s", self.dataset_root)
            return per_cat

        for cat_dir in sorted(self.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            dir_name = cat_dir.name
            if dir_name not in self.wnid_to_class:
                continue
            paths = []
            for image_path in sorted(cat_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self.IMAGE_EXTS:
                    continue
                paths.append(
                    image_path.relative_to(self.dataset_root).as_posix()
                )
            if paths:
                per_cat[dir_name] = paths
        return per_cat

    def iter_samples(self):
        """Yield samples in round-robin order across all categories.

        This guarantees that even with a small ``max_samples`` cap, the
        pipeline sees samples from every category early on, avoiding
        split-imbalance when only a fraction of the dataset is consumed.
        """
        per_cat = self._collect_per_category()
        if not per_cat:
            return

        # Build iterators for each category.
        iterators = {cat: iter(paths) for cat, paths in per_cat.items()}
        active_cats = list(sorted(iterators.keys()))

        while active_cats:
            next_round_cats = []
            for cat in active_cats:
                image_relpath = next(iterators[cat], None)
                if image_relpath is None:
                    continue
                next_round_cats.append(cat)

                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=cat,
                    class_name=self.wnid_to_class[cat],
                    extra_meta={
                        "dataset": "aid",
                        "category_dir": cat,
                    },
                )
            active_cats = next_round_cats
