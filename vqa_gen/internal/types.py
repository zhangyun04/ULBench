from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CanonicalSample:
    image_relpath: str
    wnid: str
    class_name: str
    extra_meta: Optional[dict] = None


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
