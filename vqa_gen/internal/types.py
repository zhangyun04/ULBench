from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PrebuiltQA:
    """Prebuilt VQA data provided directly by the dataset.

    When present on a :class:`CanonicalSample`, the pipeline skips
    distractor sampling and question generation, using these fields
    directly.
    """

    question: str
    choices: list[str]
    answer_index: int
    concept_axis: str = "spatial"
    forgetting_level: str = "attribute"


@dataclass(frozen=True)
class CanonicalSample:
    image_relpath: str
    wnid: str
    class_name: str
    extra_meta: Optional[dict] = None
    prebuilt_qa: Optional[PrebuiltQA] = None


@dataclass
class DatasetItem:
    id: str
    image: str
    question: str
    choices: list[str]
    answer_index: int
    forgetting_level: str
    concept_axis: str
    target_split: str
    meta: dict
