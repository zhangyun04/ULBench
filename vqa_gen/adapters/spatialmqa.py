"""SpatialMQA adapter for the spatial-reasoning VQA pipeline.

Loads the ``liuziyan/SpatialMQA`` HuggingFace dataset and yields
:class:`CanonicalSample` objects with prebuilt QA (question, options,
answer) so that the shared pipeline can skip distractor generation.

The dataset images are COCO filenames; the adapter resolves them
against a configurable image root directory.
"""

import hashlib
import logging
import random
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample, PrebuiltQA

logger = logging.getLogger(__name__)

_SPLIT_RENAME = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


class SpatialMQAAdapter:
    """Adapter for the liuziyan/SpatialMQA HuggingFace dataset.

    Parameters
    ----------
    hf_repo : str
        HuggingFace dataset identifier (default ``liuziyan/SpatialMQA``).
    image_root : str
        Root directory that contains the COCO images referenced by the
        dataset (e.g. ``data/coco/val2017``).  Image filenames from the
        dataset are resolved relative to this directory.
    splits : list[str]
        HuggingFace split names to load (e.g. ``["train", "validation", "test"]``).
    """

    def __init__(self, hf_repo="liuziyan/SpatialMQA", image_root=".", splits=None):
        self.hf_repo = hf_repo
        self.image_root = Path(image_root)
        self.splits = splits or ["train", "validation", "test"]

        # Populated during iter_samples; used by the pipeline for
        # class-level splitting and superclass lookup.
        self.wnid_to_class: dict[str, str] = {}
        self.wnid_to_superclass: dict[str, str] = {}

        self._datasets: dict[str, object] = {}
        self._load_splits()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_splits(self):
        from datasets import load_dataset

        for split in self.splits:
            try:
                ds = load_dataset(self.hf_repo, split=split)
                self._datasets[split] = ds
                logger.info(
                    "Loaded SpatialMQA split '%s': %d samples", split, len(ds)
                )
            except Exception:
                logger.warning("Could not load split '%s', skipping.", split, exc_info=True)

        self._build_class_index()

    def _build_class_index(self):
        """Pre-populate wnid_to_class by scanning all loaded splits."""
        for ds in self._datasets.values():
            for row in ds:
                wnid = self._stable_wnid(row["question"])
                if wnid not in self.wnid_to_class:
                    self.wnid_to_class[wnid] = "spatial"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    # Global pool of common spatial-reasoning distractors used to pad
    # samples that have fewer than 4 options.
    _SPATIAL_DISTRACTOR_POOL = [
        "left", "right", "above", "below", "in front of", "behind",
        "on", "under", "next to", "inside", "outside", "between",
        "near", "far", "top", "bottom", "center", "middle",
        "yes", "no", "same", "different",
        "closer", "farther", "overlapping", "adjacent",
        "north", "south", "east", "west",
    ]

    _NUM_CHOICES = 4

    @staticmethod
    def _answer_index(options: list[str], answer: str) -> int:
        """Return the index of *answer* in *options*, or -1 if not found."""
        for i, opt in enumerate(options):
            if opt.strip() == answer.strip():
                return i
        return -1

    @staticmethod
    def _stable_wnid(question: str) -> str:
        """Derive a short deterministic key from the question text.

        SpatialMQA has no class taxonomy; we use a hash of the question
        as a pseudo-wnid so that every unique question maps to its own
        'class' for splitting purposes.
        """
        h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:10]
        return f"spatialq_{h}"

    @classmethod
    def _normalize_to_4(cls, options: list[str], answer_idx: int, seed: int):
        """Normalize *options* to exactly 4 choices.

        * **> 4 options**: keep the correct answer, randomly sample 3
          from the remaining options, then shuffle.
        * **< 4 options**: pad with distractors drawn from a global pool
          (skipping duplicates), then shuffle.
        * **== 4 options**: return as-is.

        Returns ``(choices, new_answer_index)`` or ``None`` if padding
        cannot produce 4 unique choices.
        """
        n = len(options)
        answer_text = options[answer_idx]
        rng = random.Random(seed)

        if n == cls._NUM_CHOICES:
            return list(options), answer_idx

        if n > cls._NUM_CHOICES:
            # Keep answer, sample 3 from rest.
            rest = [opt for i, opt in enumerate(options) if i != answer_idx]
            sampled = rng.sample(rest, cls._NUM_CHOICES - 1)
            choices = [answer_text] + sampled
            rng.shuffle(choices)
            new_idx = choices.index(answer_text)
            return choices, new_idx

        # n < _NUM_CHOICES – pad from the distractor pool.
        existing_lower = {opt.strip().lower() for opt in options}
        candidates = [
            d for d in cls._SPATIAL_DISTRACTOR_POOL
            if d.lower() not in existing_lower
        ]
        need = cls._NUM_CHOICES - n
        if len(candidates) < need:
            return None  # cannot pad to 4 unique choices
        pad = rng.sample(candidates, need)
        choices = list(options) + pad
        rng.shuffle(choices)
        new_idx = choices.index(answer_text)
        return choices, new_idx

    # ------------------------------------------------------------------
    # Sample iteration
    # ------------------------------------------------------------------

    def iter_samples(self):
        for split in self.splits:
            ds = self._datasets.get(split)
            if ds is None:
                continue

            split_label = _SPLIT_RENAME.get(split, split)

            for row in ds:
                image_name = row["image"]
                question = row["question"]
                options = row["options"]
                answer = row["answer"]

                answer_idx = self._answer_index(options, answer)
                if answer_idx < 0:
                    logger.debug(
                        "Answer '%s' not in options %s – skipping.", answer, options
                    )
                    continue

                # Resolve image path relative to image_root.
                image_relpath = f"{split_label}/{image_name}"

                # Deterministic seed per sample.
                seed = int(
                    hashlib.sha1(image_relpath.encode("utf-8")).hexdigest()[:8], 16
                )

                result = self._normalize_to_4(options, answer_idx, seed)
                if result is None:
                    logger.debug(
                        "Cannot normalize to 4 choices for '%s' – skipping.",
                        image_relpath,
                    )
                    continue
                choices, new_answer_idx = result

                wnid = self._stable_wnid(question)
                class_name = "spatial"  # generic; not used for distractor logic
                self.wnid_to_class[wnid] = class_name

                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=wnid,
                    class_name=class_name,
                    extra_meta={
                        "dataset": "spatialmqa",
                        "original_split": split,
                        "question_text": question,
                        "answer_text": answer,
                    },
                    prebuilt_qa=PrebuiltQA(
                        question=question,
                        choices=choices,
                        answer_index=new_answer_idx,
                        concept_axis="spatial",
                        forgetting_level="scene",
                    ),
                )
