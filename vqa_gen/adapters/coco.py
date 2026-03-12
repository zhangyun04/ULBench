"""COCO 2017 adapter for the identity VQA pipeline.

Loads instance annotations, applies a single-label selection policy,
and yields CanonicalSample objects compatible with the shared pipeline.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample

logger = logging.getLogger(__name__)


class COCOAdapter:
    """Adapter for COCO-style instance-annotated datasets.

    Parameters
    ----------
    coco_root : str
        Path to the COCO dataset root directory.
    splits : list[str]
        Split names to process (e.g. ``["train", "val"]``).
    annotations : dict[str, str]
        Mapping from split name to annotation JSON path relative to *coco_root*.
    images : dict[str, str]
        Mapping from split name to image directory name relative to *coco_root*.
    policy : dict
        Single-label selection policy knobs.
    """

    def __init__(self, coco_root, splits, annotations, images, policy=None):
        self.coco_root = Path(coco_root)
        self.splits = splits
        self.annotations_map = annotations
        self.images_map = images

        policy = policy or {}
        self._require_single = policy.get("require_single_distinct_category", True)
        self._min_area_ratio = float(policy.get("min_area_ratio", 0.02))
        self._max_instances = policy.get("max_instances_total", None)

        self._categories: dict[int, dict] = {}
        self.wnid_to_class: dict[str, str] = {}
        self.wnid_to_superclass: dict[str, str] = {}

        self._split_data: dict[str, tuple] = {}

        self._load_all_splits()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _load_all_splits(self):
        """Parse every annotation file once; store indices for iteration."""
        for split in self.splits:
            ann_path = self.coco_root / self.annotations_map[split]
            if not ann_path.exists():
                logger.warning("Annotation file not found, skipping split %s: %s", split, ann_path)
                continue

            with ann_path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for cat in data.get("categories", []):
                cid = cat["id"]
                name = cat["name"]
                supercategory = cat.get("supercategory", "unknown")
                self._categories[cid] = {"name": name, "supercategory": supercategory}
                self.wnid_to_class[name] = name
                self.wnid_to_superclass[name] = supercategory

            images_info: dict[int, dict] = {}
            for img in data["images"]:
                images_info[img["id"]] = {
                    "file_name": img["file_name"],
                    "width": img["width"],
                    "height": img["height"],
                }

            anns_by_image: dict[int, list] = defaultdict(list)
            for ann in data["annotations"]:
                anns_by_image[ann["image_id"]].append(ann)

            self._split_data[split] = (images_info, anns_by_image)
            del data

        logger.info(
            "Loaded %d COCO categories across %d split(s)",
            len(self.wnid_to_class),
            len(self._split_data),
        )

    # ------------------------------------------------------------------
    # Single-label selection policy
    # ------------------------------------------------------------------

    def _apply_policy(self, anns, image_area):
        """Apply the configured selection policy to one image's annotations.

        Returns ``(category_id, selected_ann, categories_present)`` or
        ``None`` if the image should be skipped.
        """
        if not anns:
            return None

        anns = [a for a in anns if not a.get("iscrowd", 0)]
        if not anns:
            return None

        if self._max_instances is not None and len(anns) > self._max_instances:
            return None

        cat_ids = {a["category_id"] for a in anns}

        if self._require_single:
            if len(cat_ids) != 1:
                return None
            chosen_cat_id = next(iter(cat_ids))
        else:
            # Fallback (Policy B): pick the category of the largest instance.
            anns_sorted = sorted(anns, key=lambda a: a["area"], reverse=True)
            chosen_cat_id = anns_sorted[0]["category_id"]

        chosen_anns = [a for a in anns if a["category_id"] == chosen_cat_id]
        if not any(a["area"] / image_area >= self._min_area_ratio for a in chosen_anns):
            return None

        selected = max(chosen_anns, key=lambda a: a["area"])

        categories_present = sorted(
            {self._categories[cid]["name"] for cid in cat_ids if cid in self._categories}
        )

        return chosen_cat_id, selected, categories_present

    # ------------------------------------------------------------------
    # Sample iteration
    # ------------------------------------------------------------------

    def iter_samples(self):
        for split in self.splits:
            if split not in self._split_data:
                continue
            images_info, anns_by_image = self._split_data[split]
            images_dir = self.images_map[split]

            for image_id in sorted(images_info):
                img = images_info[image_id]
                image_area = img["width"] * img["height"]
                anns = anns_by_image.get(image_id, [])

                result = self._apply_policy(anns, image_area)
                if result is None:
                    continue

                cat_id, selected_ann, categories_present = result
                cat_name = self._categories[cat_id]["name"]

                image_relpath = f"{images_dir}/{img['file_name']}"

                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=cat_name,
                    class_name=cat_name,
                    extra_meta={
                        "dataset": "coco",
                        "coco_image_id": image_id,
                        "coco_category_id": cat_id,
                        "categories_present": categories_present,
                        "bbox": selected_ann.get("bbox"),
                    },
                )
