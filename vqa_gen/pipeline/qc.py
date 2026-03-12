from dataclasses import asdict

from vqa_gen.internal.types import DatasetItem
from vqa_gen.ontology.normalize import normalize_choice_text

FORBIDDEN_TOKENS = ("n0", "/", "_")
MAX_CHOICE_CHARS = 40
_SEEN_IDS = set()


def reset_qc_state():
    _SEEN_IDS.clear()


def _is_utf8(s):
    try:
        s.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def validate_item(item: DatasetItem):
    reasons = []

    if item.id in _SEEN_IDS:
        reasons.append("duplicate_id")

    if len(item.choices) != 4:
        reasons.append("invalid_choice_count")

    normalized_choices = []
    for choice in item.choices:
        if isinstance(choice, str):
            normalized_choices.append(normalize_choice_text(choice))
        else:
            normalized_choices.append("")

    if len(set(normalized_choices)) != len(normalized_choices):
        reasons.append("duplicate_choices_case_insensitive")

    if item.answer_index < 0 or item.answer_index >= len(item.choices):
        reasons.append("invalid_answer_index")
    gt_class_name = item.meta.get("class_name", "") if isinstance(item.meta, dict) else ""
    if normalize_choice_text(gt_class_name) not in normalized_choices:
        reasons.append("correct_answer_not_in_choices")

    for choice in item.choices:
        if not isinstance(choice, str):
            reasons.append("choice_not_string")
            continue
        if choice.strip() == "":
            reasons.append("empty_choice")
        if len(choice) > MAX_CHOICE_CHARS:
            reasons.append("choice_too_long")
        lowered = choice.lower()
        if any(token in lowered for token in FORBIDDEN_TOKENS):
            reasons.append("choice_contains_forbidden_token")
        if not _is_utf8(choice):
            reasons.append("choice_not_utf8")

    passed = len(reasons) == 0
    if passed:
        _SEEN_IDS.add(item.id)
    return passed, sorted(set(reasons))


def qc_pass(item):
    passed, _ = validate_item(item)
    return passed


def serialize_reject(item: DatasetItem, reasons):
    return {
        "reasons": reasons,
        "item": asdict(item),
    }
