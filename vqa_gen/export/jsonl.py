import json
from dataclasses import asdict
from pathlib import Path

from vqa_gen.internal.types import DatasetItem


def _public_item_payload(item: DatasetItem) -> dict:
    """Public JSON schema only; no extra fields."""
    return {
        "id": item.id,
        "image": item.image,
        "question": item.question,
        "choices": item.choices,
        "answer_index": item.answer_index,
        "forgetting_level": item.forgetting_level,
        "concept_axis": item.concept_axis,
        "target_split": item.target_split,
        "meta": item.meta,
    }


def write_jsonl(path, rows, *, public_schema_only=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if public_schema_only and isinstance(row, DatasetItem):
                payload = _public_item_payload(row)
            elif hasattr(row, "__dataclass_fields__"):
                payload = asdict(row)
            else:
                payload = row
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
