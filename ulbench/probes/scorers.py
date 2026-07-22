"""Registered scorers for direct probes (spec §6.5, §7.2, §10.1).

Every scorer returns exactly one response status and never collapses
refusal / invalid_format into ``incorrect``. Semantic-judge scoring is
deliberately absent: it may not gate eligibility or main conclusions before
its human-agreement validation passes G0 (frozen decision, spec §7.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from vqa_gen.ontology.normalize import normalize_choice_text

from ulbench.schema import BenchmarkItem, QuestionFormat, ResponseStatus

# Versioned, conservative refusal lexicon. Matches are refusals only when the
# output contains no scorable answer signal; parseable answers win.
REFUSAL_LEXICON_VERSION = "refusal_lexicon.v1"
_REFUSAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bI (?:cannot|can't|can not|won't|will not|refuse to)\b",
        r"\bI(?:'m| am) (?:not able|unable)\b",
        r"\b(?:sorry|apologi[sz]e)\b.{0,40}\b(?:cannot|can't|unable)\b",
        r"\bnot (?:allowed|permitted) to\b",
        r"\bI must decline\b",
        r"\bas an AI\b.{0,40}\b(?:cannot|can't)\b",
    )
]


@dataclass
class ScoreResult:
    response_status: ResponseStatus
    prediction: Optional[str]
    scorer_id: str
    scorer_version: str
    detail: dict[str, Any] = field(default_factory=dict)


def detect_refusal(raw_output: str) -> Optional[str]:
    """Return the matched refusal pattern, or None."""
    for pattern in _REFUSAL_PATTERNS:
        match = pattern.search(raw_output)
        if match:
            return match.group(0)
    return None


def score_mcq(raw_output: str, item: BenchmarkItem) -> ScoreResult:
    """Exact option correctness for MCQ (spec §7.2).

    Parse order: first in-range standalone digit; otherwise a normalized
    full-output match against the option texts. Refusal wins only when no
    answer is parseable.
    """
    if item.question_format != QuestionFormat.MCQ or item.choices is None:
        raise ValueError(f"{item.item_id} is not an mcq item")
    scorer = dict(scorer_id="mcq_exact", scorer_version="1.0.0")
    text = raw_output.strip()

    choice_count = len(item.choices)
    digit_match = re.search(rf"\b([0-{choice_count - 1}])\b", text)
    predicted_index: Optional[int] = None
    match_mode = None
    if digit_match:
        predicted_index = int(digit_match.group(1))
        match_mode = "digit"
    else:
        normalized_output = normalize_choice_text(text)
        for index, choice in enumerate(item.choices):
            if normalized_output == normalize_choice_text(choice):
                predicted_index = index
                match_mode = "choice_text"
                break

    if predicted_index is None:
        refusal = detect_refusal(text)
        if refusal:
            return ScoreResult(
                ResponseStatus.REFUSAL, None, **scorer,
                detail={"refusal_match": refusal,
                        "lexicon": REFUSAL_LEXICON_VERSION},
            )
        return ScoreResult(
            ResponseStatus.INVALID_FORMAT, None, **scorer,
            detail={"reason": "no in-range option index or option text"},
        )

    status = (ResponseStatus.CORRECT if predicted_index == item.answer_index
              else ResponseStatus.INCORRECT)
    return ScoreResult(
        status, str(predicted_index), **scorer,
        detail={"match_mode": match_mode},
    )


def score_short_answer(raw_output: str, item: BenchmarkItem) -> ScoreResult:
    """Exact / normalized alias matching for short answer (spec §7.2).

    ``detail`` reports both match levels separately so tables never merge
    scorers silently; ``correct`` means the normalized match succeeded.
    """
    if item.question_format != QuestionFormat.SHORT_ANSWER:
        raise ValueError(f"{item.item_id} is not a short-answer item")
    scorer = dict(scorer_id="short_answer_alias", scorer_version="1.0.0")
    text = raw_output.strip()

    if not text:
        return ScoreResult(
            ResponseStatus.INVALID_FORMAT, None, **scorer,
            detail={"reason": "empty output"},
        )

    stripped = text.strip().strip(".,!?\"'()")
    exact = any(stripped == answer for answer in item.accepted_answers)
    normalized_output = normalize_choice_text(stripped)
    normalized_match = None
    for answer in item.accepted_answers:
        if normalized_output == normalize_choice_text(answer):
            normalized_match = answer
            break

    if normalized_match is not None:
        return ScoreResult(
            ResponseStatus.CORRECT, stripped, **scorer,
            detail={"exact_match": exact, "normalized_match": normalized_match},
        )

    refusal = detect_refusal(text)
    if refusal:
        return ScoreResult(
            ResponseStatus.REFUSAL, None, **scorer,
            detail={"refusal_match": refusal,
                    "lexicon": REFUSAL_LEXICON_VERSION},
        )
    return ScoreResult(
        ResponseStatus.INCORRECT, stripped, **scorer,
        detail={"exact_match": False, "normalized_match": None},
    )
