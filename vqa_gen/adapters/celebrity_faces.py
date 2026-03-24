"""Celebrity Face Image Dataset adapter for the person-identity VQA pipeline.

Loads the Celebrity Faces dataset from a local directory organized as
``<root>/Person Name/image.jpg`` and yields :class:`CanonicalSample`
objects compatible with the shared pipeline.

The dataset has no official train/test split; all images are iterated
in round-robin order across celebrities to ensure balanced coverage
even under a ``max_samples`` cap.

Reference: https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset
"""

import json
import logging
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample

logger = logging.getLogger(__name__)


class CelebrityFacesAdapter:
    """Adapter for the Celebrity Face Image Dataset.

    Parameters
    ----------
    dataset_root : str
        Root directory containing one sub-directory per celebrity.
    category_metadata_path : str, optional
        Path to JSON file mapping directory names to
        ``{"display_name": ..., "supercategory": ...}``.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, dataset_root, category_metadata_path=None):
        self.dataset_root = Path(dataset_root)

        self.wnid_to_class: dict[str, str] = {}
        self.wnid_to_superclass: dict[str, str] = {}

        if category_metadata_path:
            self._load_category_metadata(category_metadata_path)
        else:
            self._discover_categories()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_category_metadata(self, path):
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        for dir_name, info in data.items():
            display = info.get("display_name", dir_name)
            supercat = info.get("supercategory", "unknown")
            self.wnid_to_class[dir_name] = display
            self.wnid_to_superclass[dir_name] = supercat

        logger.info(
            "Loaded %d celebrity categories from %s",
            len(self.wnid_to_class), path,
        )

    def _discover_categories(self):
        if not self.dataset_root.exists():
            logger.warning("Dataset root not found: %s", self.dataset_root)
            return
        for cat_dir in sorted(self.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            name = cat_dir.name
            self.wnid_to_class[name] = name
            self.wnid_to_superclass[name] = "unknown"

        logger.info(
            "Discovered %d celebrities from filesystem",
            len(self.wnid_to_class),
        )

    # ------------------------------------------------------------------
    # Round-robin iteration
    # ------------------------------------------------------------------

    def _collect_per_category(self):
        per_cat: dict[str, list[str]] = {}
        if not self.dataset_root.exists():
            return per_cat

        for cat_dir in sorted(self.dataset_root.iterdir()):
            if not cat_dir.is_dir():
                continue
            name = cat_dir.name
            if name not in self.wnid_to_class:
                continue
            paths = []
            for img in sorted(cat_dir.rglob("*")):
                if img.is_file() and img.suffix.lower() in self.IMAGE_EXTS:
                    paths.append(
                        img.relative_to(self.dataset_root).as_posix()
                    )
            if paths:
                per_cat[name] = paths
        return per_cat

    def iter_samples(self):
        """Yield samples in round-robin order across celebrities."""
        per_cat = self._collect_per_category()
        if not per_cat:
            return

        iterators = {cat: iter(paths) for cat, paths in per_cat.items()}
        active = sorted(iterators.keys())

        while active:
            next_round = []
            for cat in active:
                image_relpath = next(iterators[cat], None)
                if image_relpath is None:
                    continue
                next_round.append(cat)

                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=cat,
                    class_name=self.wnid_to_class[cat],
                    extra_meta={
                        "dataset": "celebrity_faces",
                        "person_name": cat,
                    },
                )
            active = next_round
