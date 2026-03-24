"""Logo-2K+ adapter for the logo-identity VQA pipeline.

Loads the Logo-2K+ dataset from a local directory organized as
``<root>/SuperCategory/LogoClass/image.jpg`` and yields
:class:`CanonicalSample` objects compatible with the shared pipeline.

The two-level hierarchy (10 root categories → 2341 logo classes)
naturally provides supercategory grouping for hard-negative sampling.

The dataset has no official train/test split; all images are iterated
in round-robin order across logo classes to ensure balanced coverage
even under a ``max_samples`` cap.

Reference: Wang et al., "Logo-2K+: A Large-Scale Logo Dataset for
Scalable Logo Classification", AAAI 2020.
"""

import logging
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample

logger = logging.getLogger(__name__)


class Logo2KPlusAdapter:
    """Adapter for the Logo-2K+ dataset.

    Parameters
    ----------
    dataset_root : str
        Root directory of the Logo-2K+ dataset (contains supercategory
        sub-directories like ``Food/``, ``Clothes/``, etc.).
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, dataset_root):
        self.dataset_root = Path(dataset_root)

        self.wnid_to_class: dict[str, str] = {}
        self.wnid_to_superclass: dict[str, str] = {}

        self._discover_hierarchy()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _discover_hierarchy(self):
        """Walk the two-level directory tree to discover all logo classes
        and their supercategories."""
        if not self.dataset_root.exists():
            logger.warning("Dataset root not found: %s", self.dataset_root)
            return

        for supercat_dir in sorted(self.dataset_root.iterdir()):
            if not supercat_dir.is_dir():
                continue
            supercat_name = supercat_dir.name

            for logo_dir in sorted(supercat_dir.iterdir()):
                if not logo_dir.is_dir():
                    continue
                logo_name = logo_dir.name
                # Use "supercat/logo" as wnid to guarantee uniqueness
                # (different supercategories could theoretically share
                # a logo directory name).
                wnid = f"{supercat_name}/{logo_name}"
                display_name = logo_name.replace("_", " ")
                self.wnid_to_class[wnid] = display_name
                self.wnid_to_superclass[wnid] = supercat_name

        logger.info(
            "Discovered %d logo classes across %d supercategories",
            len(self.wnid_to_class),
            len({v for v in self.wnid_to_superclass.values()}),
        )

    # ------------------------------------------------------------------
    # Round-robin iteration
    # ------------------------------------------------------------------

    def _collect_per_class(self):
        """Return ``{wnid: [image_relpath, ...]}``, sorted within each class."""
        per_class: dict[str, list[str]] = {}
        if not self.dataset_root.exists():
            return per_class

        for supercat_dir in sorted(self.dataset_root.iterdir()):
            if not supercat_dir.is_dir():
                continue
            for logo_dir in sorted(supercat_dir.iterdir()):
                if not logo_dir.is_dir():
                    continue
                wnid = f"{supercat_dir.name}/{logo_dir.name}"
                if wnid not in self.wnid_to_class:
                    continue
                paths = []
                for img in sorted(logo_dir.rglob("*")):
                    if img.is_file() and img.suffix.lower() in self.IMAGE_EXTS:
                        paths.append(
                            img.relative_to(self.dataset_root).as_posix()
                        )
                if paths:
                    per_class[wnid] = paths
        return per_class

    def iter_samples(self):
        """Yield samples in round-robin order across all logo classes."""
        per_class = self._collect_per_class()
        if not per_class:
            return

        iterators = {cls: iter(paths) for cls, paths in per_class.items()}
        active = sorted(iterators.keys())

        while active:
            next_round = []
            for cls in active:
                image_relpath = next(iterators[cls], None)
                if image_relpath is None:
                    continue
                next_round.append(cls)

                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=cls,
                    class_name=self.wnid_to_class[cls],
                    extra_meta={
                        "dataset": "logo2kplus",
                        "supercategory": self.wnid_to_superclass[cls],
                    },
                )
            active = next_round
