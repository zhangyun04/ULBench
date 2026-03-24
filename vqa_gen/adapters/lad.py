"""LAD (Large-scale Attribute Dataset) adapter for attribute-based VQA.

Loads the LAD dataset and generates attribute VQA by reading class-level
attribute annotations from ``attributes_per_class.txt``.  The adapter is
**generic**: the ``attribute_group`` parameter selects which attribute
category to ask about (color, shape, size, habitat, behaviour, …).

For each image the question asks about the object's dominant attribute
value; the correct answer is the highest-scoring value and three
distractors are sampled from the remaining values.

Reference: Zhao et al., "A Large-scale Attribute Dataset for Zero-shot
Learning", 2018.
"""

import hashlib
import logging
import random
import re
from collections import defaultdict
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample, PrebuiltQA

logger = logging.getLogger(__name__)

# ---- Domain offsets inside the 359-dim attribute vector ----
_DOMAIN_OFFSETS = {
    "A": 0,
    "F": 123,
    "V": 181,
    "E": 262,
    "H": 337,
}

_DOMAIN_DISPLAY = {
    "A": "animal",
    "F": "fruit",
    "V": "vehicle",
    "E": "electronic device",
    "H": "hairstyle",
}

# ---- Question templates per attribute group ----
_QUESTION_TEMPLATES = {
    "color":      "What color is the {class_name} shown in the image?",
    "outside color": "What color is the {class_name} shown in the image?",
    "shape":      "What shape is the {class_name} shown in the image?",
    "size":       "What size is the {class_name} shown in the image?",
    "habitat":    "Where does the {class_name} shown in the image live?",
    "behaviour":  "What can the {class_name} shown in the image do?",
    "marking":    "What marking does the {class_name} shown in the image have?",
    "taste":      "What does the {class_name} shown in the image taste like?",
    "appearance": "What does the {class_name} shown in the image look like?",
    "diet":       "What does the {class_name} shown in the image eat?",
}
_DEFAULT_QUESTION = "Which attribute best describes the {class_name} shown in the image?"

# ---- Concept axis names per attribute group ----
_CONCEPT_AXIS = {
    "color": "attribute_color",
    "outside color": "attribute_color",
    "shape": "attribute_shape",
    "size": "attribute_size",
    "habitat": "attribute_habitat",
    "behaviour": "attribute_behaviour",
    "marking": "attribute_marking",
    "taste": "attribute_taste",
    "appearance": "attribute_appearance",
    "diet": "attribute_diet",
}

_NUM_CHOICES = 4

# Fallback distractor pools for attribute groups where the total number
# of distinct values across all domains is fewer than _NUM_CHOICES.
_EXTRA_DISTRACTORS = {
    "size": ["tiny", "very large", "huge", "medium-large"],
}



def _extract_short_label(description: str) -> str:
    """Extract a short choice label from an attribute description.

    Examples::

        "is brown"                     → "brown"
        "is small (compared to pigs)"  → "small"
        "lives in the forest"          → "the forest"
        "can swim"                     → "swim"
        "has stripes"                  → "stripes"
        "eats meat"                    → "meat"
    """
    desc = description.strip()
    # "is X (compared to ...)" → "X"
    m = re.match(r"is\s+(.+?)\s*\(", desc)
    if m:
        return m.group(1).strip()
    # "is X" → "X"
    if desc.startswith("is "):
        return desc[3:].strip()
    # "lives in X" / "lives on X"
    m = re.match(r"lives\s+(?:in|on)\s+(.+)", desc)
    if m:
        return m.group(1).strip()
    # "can X"
    if desc.startswith("can "):
        return desc[4:].strip()
    # "has X" / "have X"
    m = re.match(r"(?:has|have)\s+(.+)", desc)
    if m:
        return m.group(1).strip()
    # "eats X"
    if desc.startswith("eats "):
        return desc[5:].strip()
    # "tastes X"
    if desc.startswith("tastes "):
        return desc[7:].strip()
    # "moves X"
    if desc.startswith("moves "):
        return desc[6:].strip()
    # fallback
    return desc


class LADAdapter:
    """Generic adapter for LAD attribute VQA tasks.

    Parameters
    ----------
    dataset_root : str
        Root of the LAD dataset.
    attribute_group : str
        Which attribute category to query.  Must match a group name in
        ``attribute_list.txt`` (e.g. ``"color"``, ``"shape"``,
        ``"size"``, ``"habitat"``, ``"behaviour"``).
    concept_axis : str or None
        Value for ``concept_axis`` in the output.  If ``None``, derived
        automatically from *attribute_group*.
    domains : list[str] or None
        Restrict to these domains.  ``None`` = all that have the group.
    threshold : float
        Minimum class-level score to consider a value as "present".
    """

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        dataset_root,
        attribute_group="color",
        concept_axis=None,
        domains=None,
        threshold=0.3,
    ):
        self.dataset_root = Path(dataset_root)
        self.attribute_group = attribute_group.lower()
        self.concept_axis = concept_axis or _CONCEPT_AXIS.get(
            self.attribute_group, f"attribute_{self.attribute_group}"
        )
        self._requested_domains = set(domains) if domains else None
        self.threshold = threshold

        self.wnid_to_class: dict[str, str] = {}
        self.wnid_to_superclass: dict[str, str] = {}

        # Internal.
        self._label_to_name: dict[str, str] = {}
        self._label_to_domain: dict[str, str] = {}
        # domain → [(global_vector_index, short_label)]
        self._group_attrs: dict[str, list[tuple[int, str]]] = {}
        # label_id → (value_labels, scores)
        self._class_attr_data: dict[str, tuple] = {}
        self._image_entries: list[tuple] = []

        self._parse_attribute_list()
        self._load_labels()
        self._load_class_attributes()
        self._load_images()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _parse_attribute_list(self):
        """Parse ``attribute_list.txt`` and index attributes for the
        requested group across all domains."""
        path = self.dataset_root / "attribute_list.txt"
        # Track the running index within each domain so we can compute
        # the absolute index into the 359-dim vector.
        domain_counter: dict[str, int] = defaultdict(int)

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                attr_id = parts[0]          # e.g. Attr_A_005
                desc_full = parts[1]        # e.g. "color: is brown"
                domain = attr_id.split("_")[1]
                local_idx = domain_counter[domain]
                domain_counter[domain] += 1

                # Split "group: value_description"
                if ": " not in desc_full:
                    continue
                group, value_desc = desc_full.split(": ", 1)
                group = group.strip().lower()

                if group != self.attribute_group:
                    continue

                global_idx = _DOMAIN_OFFSETS[domain] + local_idx
                short_label = _extract_short_label(value_desc)
                self._group_attrs.setdefault(domain, []).append(
                    (global_idx, short_label)
                )

        # Filter domains.
        if self._requested_domains:
            self._group_attrs = {
                d: v for d, v in self._group_attrs.items()
                if d in self._requested_domains
            }

        active_domains = sorted(self._group_attrs.keys())
        n_total = sum(len(v) for v in self._group_attrs.values())
        logger.info(
            "Attribute group '%s': %d values across domains %s",
            self.attribute_group, n_total, active_domains,
        )
        if not self._group_attrs:
            logger.warning(
                "No attributes found for group '%s'. "
                "Available groups can be listed from attribute_list.txt.",
                self.attribute_group,
            )

        # Build a global pool of unique labels across ALL domains for this
        # group.  Used as fallback when a single domain has fewer than
        # _NUM_CHOICES distinct values (e.g. size has only 3).
        seen_lower: set[str] = set()
        self._global_label_pool: list[str] = []
        for entries in self._group_attrs.values():
            for _, lbl in entries:
                if lbl.lower() not in seen_lower:
                    self._global_label_pool.append(lbl)
                    seen_lower.add(lbl.lower())

    def _load_labels(self):
        """Parse ``label_list.txt`` (only classes in active domains)."""
        active_domains = set(self._group_attrs.keys())
        path = self.dataset_root / "label_list.txt"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                label_id, name = parts[0], parts[1]
                domain = label_id.split("_")[1]
                if domain not in active_domains:
                    continue
                self._label_to_name[label_id] = name
                self._label_to_domain[label_id] = domain
                self.wnid_to_class[name] = name
                self.wnid_to_superclass[name] = _DOMAIN_DISPLAY.get(domain, domain)

        logger.info("Loaded %d LAD classes for group '%s'",
                     len(self._label_to_name), self.attribute_group)

    def _load_class_attributes(self):
        """Parse ``attributes_per_class.txt`` and extract scores for
        the selected attribute group."""
        path = self.dataset_root / "attributes_per_class.txt"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                label_id = line.split(",")[0].strip()
                if label_id not in self._label_to_name:
                    continue
                domain = self._label_to_domain[label_id]
                attr_entries = self._group_attrs.get(domain)
                if not attr_entries:
                    continue

                match = re.search(r"\[(.+)\]", line)
                if not match:
                    continue
                all_vals = [float(v) for v in match.group(1).split()]

                labels = [lbl for _, lbl in attr_entries]
                scores = [all_vals[idx] for idx, _ in attr_entries]
                self._class_attr_data[label_id] = (labels, scores)

        logger.info("Loaded attribute scores for %d classes",
                     len(self._class_attr_data))

    def _load_images(self):
        """Parse ``images.txt``."""
        path = self.dataset_root / "images.txt"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                image_id, label_id, image_path = parts[0], parts[1], parts[3]
                if label_id not in self._label_to_name:
                    continue
                self._image_entries.append((image_id, label_id, image_path))

        logger.info("Loaded %d image entries", len(self._image_entries))

    # ------------------------------------------------------------------
    # Choice construction
    # ------------------------------------------------------------------

    def _build_choices(self, label_id, seed):
        """Build 4 attribute choices: 1 dominant + 3 distractors.

        Returns ``(question, choices, answer_index)`` or ``None``.
        """
        if label_id not in self._class_attr_data:
            return None

        labels, scores = self._class_attr_data[label_id]
        class_name = self._label_to_name[label_id]

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        best_idx, best_score = ranked[0]

        if best_score < self.threshold:
            return None

        answer = labels[best_idx]

        # Deduplicated distractor pool (preserving rank order).
        seen = {answer.lower()}
        distractors = []
        for i, _ in ranked[1:]:
            lbl = labels[i]
            if lbl.lower() not in seen:
                distractors.append(lbl)
                seen.add(lbl.lower())

        # Fallback 1: pad from the global cross-domain label pool.
        need = _NUM_CHOICES - 1
        if len(distractors) < need:
            for lbl in self._global_label_pool:
                if lbl.lower() not in seen:
                    distractors.append(lbl)
                    seen.add(lbl.lower())
                if len(distractors) >= need:
                    break

        # Fallback 2: pad from the extra distractor pool for groups
        # where the total distinct values < _NUM_CHOICES (e.g. size).
        if len(distractors) < need:
            extras = _EXTRA_DISTRACTORS.get(self.attribute_group, [])
            for lbl in extras:
                if lbl.lower() not in seen:
                    distractors.append(lbl)
                    seen.add(lbl.lower())
                if len(distractors) >= need:
                    break

        if len(distractors) < need:
            return None

        rng = random.Random(seed)
        chosen_distractors = distractors[:need]
        choices = [answer] + chosen_distractors
        rng.shuffle(choices)
        answer_index = choices.index(answer)

        template = _QUESTION_TEMPLATES.get(
            self.attribute_group, _DEFAULT_QUESTION
        )
        question = template.format(class_name=class_name)

        return question, choices, answer_index

    # ------------------------------------------------------------------
    # Round-robin iteration
    # ------------------------------------------------------------------

    def _collect_per_class(self):
        per_class: dict[str, list[tuple]] = {}
        for entry in self._image_entries:
            label_id = entry[1]
            per_class.setdefault(label_id, []).append(entry)
        for entries in per_class.values():
            entries.sort(key=lambda e: e[0])
        return per_class

    def iter_samples(self):
        """Yield samples in round-robin order across classes."""
        per_class = self._collect_per_class()
        if not per_class:
            return

        iterators = {cls: iter(entries) for cls, entries in per_class.items()}
        active = sorted(iterators.keys())

        while active:
            next_round = []
            for cls in active:
                entry = next(iterators[cls], None)
                if entry is None:
                    continue
                next_round.append(cls)

                image_id, label_id, image_path = entry
                class_name = self._label_to_name[label_id]
                domain = self._label_to_domain[label_id]

                seed = int(hashlib.sha1(
                    image_path.encode("utf-8")).hexdigest()[:8], 16)
                result = self._build_choices(label_id, seed)
                if result is None:
                    continue

                question, choices, answer_index = result

                yield CanonicalSample(
                    image_relpath=image_path,
                    wnid=class_name,
                    class_name=class_name,
                    extra_meta={
                        "dataset": "lad",
                        "domain": domain,
                        "domain_name": _DOMAIN_DISPLAY.get(domain, domain),
                        "lad_label_id": label_id,
                        "lad_image_id": image_id,
                        "attribute_group": self.attribute_group,
                        "answer_value": choices[answer_index],
                    },
                    prebuilt_qa=PrebuiltQA(
                        question=question,
                        choices=choices,
                        answer_index=answer_index,
                        concept_axis=self.concept_axis,
                        forgetting_level="object",
                    ),
                )
            active = next_round
