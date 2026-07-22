"""Response-status accounting (spec §10.1, roadmap B-P0.6).

Refusal, invalid format, policy blocks, and operational errors stay separate
everywhere. Access reduction with collapsed coverage must be visible, so both
access views and the full status decomposition always travel together.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Optional

from ulbench.schema import ResponseStatus

SCORABLE_STATUSES = (ResponseStatus.CORRECT, ResponseStatus.INCORRECT)


def _as_status(value: ResponseStatus | str) -> ResponseStatus:
    return value if isinstance(value, ResponseStatus) else ResponseStatus(value)


def response_accounting(
    statuses: Iterable[ResponseStatus | str],
) -> dict[str, Any]:
    """Counts, ``access_rate``, ``conditional_accuracy``, and coverage.

    ``access_rate = correct / attempted`` (observed retrieval);
    ``conditional_accuracy = correct / scorable`` (null when nothing is
    scorable); ``coverage = scorable / attempted``.
    """
    counts = Counter(_as_status(status) for status in statuses)
    attempted = sum(counts.values())
    correct = counts[ResponseStatus.CORRECT]
    scorable = correct + counts[ResponseStatus.INCORRECT]

    def rate(count: int) -> Optional[float]:
        return count / attempted if attempted else None

    return {
        "attempted": attempted,
        "scorable": scorable,
        "counts": {status.value: counts[status] for status in ResponseStatus},
        "access_rate": rate(correct),
        "conditional_accuracy": correct / scorable if scorable else None,
        "coverage": rate(scorable),
        "refusal_rate": rate(counts[ResponseStatus.REFUSAL]),
        "invalid_format_rate": rate(counts[ResponseStatus.INVALID_FORMAT]),
        "policy_block_rate": rate(counts[ResponseStatus.POLICY_BLOCK]),
        "error_rate": rate(
            counts[ResponseStatus.API_ERROR] + counts[ResponseStatus.MODEL_ERROR]
        ),
    }


def accounting_by_group(
    records: Iterable[dict[str, Any]],
    group_keys: tuple[str, ...],
    *,
    status_key: str = "response_status",
) -> dict[tuple, dict[str, Any]]:
    """Group records by *group_keys* and account each group separately.

    Metrics must be reported by model/method/concept/probe/condition before
    any macro summary (spec §10) — this is the grouping primitive for that.
    """
    grouped: dict[tuple, list] = {}
    for record in records:
        key = tuple(record[key] for key in group_keys)
        grouped.setdefault(key, []).append(record[status_key])
    return {key: response_accounting(statuses)
            for key, statuses in sorted(grouped.items(), key=lambda kv: repr(kv[0]))}
