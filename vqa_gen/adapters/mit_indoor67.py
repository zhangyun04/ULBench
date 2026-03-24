"""MIT Indoor-67 adapter for the scene-type VQA pipeline.

Loads the MIT Indoor-67 dataset from a local directory, resolves
train/test splits from the official split files (``TrainImages.txt``,
``TestImages.txt``), and yields :class:`CanonicalSample` objects
compatible with the shared pipeline.

Expected directory layout::

    <dataset_root>/
        Images/
            airport_inside/
                airport_inside_0001.jpg
                ...
            bakery/
                ...
        TrainImages.txt
        TestImages.txt

Reference: https://web.mit.edu/torralba/www/indoor.html
"""

import json
import logging
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample

logger = logging.getLogger(__name__)


class MITIndoor67Adapter:
    """Adapter for the MIT Indoor-67 scene recognition dataset.

    Parameters
    ----------
    dataset_root : str
        Root directory of the MIT Indoor-67 dataset.
    splits : list[str]
        Split names to process.  Each name must have a corresponding
        ``<Name>Images.txt`` file under *dataset_root* (e.g.
        ``"Train"`` → ``TrainImages.txt``).
    category_metadata_path : str
        Path to the JSON file that maps directory names to
        ``{"display_name": ..., "supercategory": ...}``.
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, dataset_root, splits=None, category_metadata_path=None):
        self.dataset_root = Path(dataset_root)
        self.splits = splits or ["Train", "Test"]

        # category metadata
        self._category_meta: dict[str, dict] = {}
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
        """Load the 67-category JSON with display names and supercategories."""
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        for dir_name, info in data.items():
            display = info.get("display_name", dir_name.replace("_", " "))
            supercat = info.get("supercategory", "unknown")
            self._category_meta[dir_name] = info
            self.wnid_to_class[dir_name] = display
            self.wnid_to_superclass[dir_name] = supercat

        logger.info(
            "Loaded %d indoor categories from %s", len(self.wnid_to_class), path
        )

    def _discover_categories(self):
        """Fallback: discover categories from the Images/ directory tree."""
        images_dir = self.dataset_root / "Images"
        if not images_dir.exists():
            logger.warning("Images/ directory not found at %s", images_dir)
            return

        for cat_dir in sorted(images_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            dir_name = cat_dir.name
            display = dir_name.replace("_", " ")
            self.wnid_to_class[dir_name] = display
            self.wnid_to_superclass[dir_name] = "unknown"

        logger.info(
            "Discovered %d indoor categories from filesystem", len(self.wnid_to_class)
        )

    # ------------------------------------------------------------------
    # Split file loading
    # ------------------------------------------------------------------

    def _load_split_file(self, split_name):
        """Read ``<SplitName>Images.txt`` and return a list of relative paths.

        Each line in the file is expected to be a path like
        ``category_name/image_file.jpg``.
        """
        filename = f"{split_name}Images.txt"
        split_path = self.dataset_root / filename
        if not split_path.exists():
            logger.warning("Split file not found: %s", split_path)
            return []

        paths = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    paths.append(line)

        logger.info("Split '%s': %d image paths loaded from %s", split_name, len(paths), filename)
        return paths

    # ------------------------------------------------------------------
    # Sample iteration
    # ------------------------------------------------------------------

    def iter_samples(self):
        for split_name in self.splits:
            rel_paths = self._load_split_file(split_name)
            split_label = split_name.lower()  # "Train" -> "train"

            for rel_path in rel_paths:
                # rel_path is like "category_name/image_file.jpg"
                parts = rel_path.split("/", 1)
                if len(parts) != 2:
                    logger.debug("Unexpected path format: %s", rel_path)
                    continue

                cat_dir_name, image_filename = parts

                # Check file extension
                ext = Path(image_filename).suffix.lower()
                if ext not in self.IMAGE_EXTS:
                    continue

                # Resolve category
                if cat_dir_name not in self.wnid_to_class:
                    logger.debug("Unknown category '%s' – skipping.", cat_dir_name)
                    continue

                class_name = self.wnid_to_class[cat_dir_name]

                # image_relpath is relative to dataset_root/Images/
                # Use split_label as prefix for consistency with other adapters
                image_relpath = f"{split_label}/{cat_dir_name}/{image_filename}"

                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=cat_dir_name,
                    class_name=class_name,
                    extra_meta={
                        "dataset": "mit_indoor67",
                        "original_split": split_name,
                        "category_dir": cat_dir_name,
                    },
                )
