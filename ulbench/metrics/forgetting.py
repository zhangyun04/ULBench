"""Forgetting Effect relative to M0 (spec §10.4) and Matched Retain Fidelity
(spec §10.5).

Both metrics pair requests by identity keys and fail loudly on misalignment —
pairs silently dropped would bias the effect. Neither number is ever named
proof of deletion.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

from ulbench.metrics.accounting import response_accounting
from ulbench.schema import ResponseStatus, SchemaValidationError

PAIR_KEYS = ("sample_id", "probe_id", "prompt_variant_id", "input_condition")


def _as_status(value: ResponseStatus | str) -> ResponseStatus:
    return value if isinstance(value, ResponseStatus) else ResponseStatus(value)


def _index_records(
    records: Iterable[dict[str, Any]], pair_keys: tuple[str, ...], label: str
) -> dict[tuple, dict[str, Any]]:
    indexed: dict[tuple, dict[str, Any]] = {}
    errors = []
    for position, record in enumerate(records):
        missing = [key for key in pair_keys if key not in record]
        if missing:
            errors.append(
                f"{label} record {position} missing keys: {', '.join(missing)}"
            )
            continue
        key = tuple(record[key] for key in pair_keys)
        if key in indexed:
            errors.append(f"{label} has duplicate pair key {key!r}")
            continue
        indexed[key] = record
    if errors:
        raise SchemaValidationError("paired metrics", errors)
    return indexed


def _delta(after: Optional[float], before: Optional[float]) -> Optional[float]:
    if after is None or before is None:
        return None
    return after - before


def forgetting_effect(
    m0_records: Iterable[dict[str, Any]],
    mu_records: Iterable[dict[str, Any]],
    *,
    pair_keys: tuple[str, ...] = PAIR_KEYS,
    status_key: str = "response_status",
) -> dict[str, Any]:
    """``FE_access`` / ``FE_clean`` with full status-delta decomposition.

    Requests must align one-to-one on *pair_keys* across states; any
    unmatched key on either side raises instead of silently shrinking the
    comparison.
    """
    m0 = _index_records(m0_records, pair_keys, "M0")
    mu = _index_records(mu_records, pair_keys, "Mu")
    only_m0 = sorted(set(m0) - set(mu))
    only_mu = sorted(set(mu) - set(m0))
    if only_m0 or only_mu:
        raise SchemaValidationError(
            "forgetting_effect",
            [f"unpaired M0 keys: {only_m0[:5]}"] * bool(only_m0)
            + [f"unpaired Mu keys: {only_mu[:5]}"] * bool(only_mu),
        )
    if not m0:
        raise SchemaValidationError("forgetting_effect", ["no paired records"])

    m0_accounting = response_accounting(
        record[status_key] for record in m0.values()
    )
    mu_accounting = response_accounting(
        record[status_key] for record in mu.values()
    )

    clean_pairs = [
        (key, _as_status(m0[key][status_key]), _as_status(mu[key][status_key]))
        for key in sorted(m0)
        if _as_status(m0[key][status_key]) in
        (ResponseStatus.CORRECT, ResponseStatus.INCORRECT)
        and _as_status(mu[key][status_key]) in
        (ResponseStatus.CORRECT, ResponseStatus.INCORRECT)
    ]
    if clean_pairs:
        m0_clean = sum(
            status == ResponseStatus.CORRECT for _, status, _ in clean_pairs
        ) / len(clean_pairs)
        mu_clean = sum(
            status == ResponseStatus.CORRECT for _, _, status in clean_pairs
        ) / len(clean_pairs)
        fe_clean = m0_clean - mu_clean
    else:
        fe_clean = None

    return {
        "pair_count": len(m0),
        "fe_access": _delta(m0_accounting["access_rate"],
                            mu_accounting["access_rate"]),
        "fe_clean": fe_clean,
        "clean_pair_count": len(clean_pairs),
        "m0": m0_accounting,
        "mu": mu_accounting,
        "deltas": {
            name: _delta(mu_accounting[name], m0_accounting[name])
            for name in (
                "coverage", "refusal_rate", "invalid_format_rate",
                "policy_block_rate", "error_rate",
            )
        },
    }


def matched_retain_fidelity(
    m0_records: Iterable[dict[str, Any]],
    mu_records: Iterable[dict[str, Any]],
    *,
    pair_keys: tuple[str, ...] = PAIR_KEYS,
    status_key: str = "response_status",
    prediction_key: str = "prediction",
) -> dict[str, Any]:
    """Paired correctness retention and output consistency on matched retain
    items (spec §10.5). Uses the same accounting as forget metrics."""
    effect = forgetting_effect(
        m0_records, mu_records, pair_keys=pair_keys, status_key=status_key
    )

    m0 = _index_records(m0_records, pair_keys, "M0")
    mu = _index_records(mu_records, pair_keys, "Mu")
    kept = lost = gained = 0
    consistent = comparable = 0
    for key in sorted(m0):
        before = _as_status(m0[key][status_key])
        after = _as_status(mu[key][status_key])
        if before == ResponseStatus.CORRECT and after == ResponseStatus.CORRECT:
            kept += 1
        elif before == ResponseStatus.CORRECT:
            lost += 1
        elif after == ResponseStatus.CORRECT:
            gained += 1
        before_prediction = m0[key].get(prediction_key)
        after_prediction = mu[key].get(prediction_key)
        if before_prediction is not None and after_prediction is not None:
            comparable += 1
            consistent += before_prediction == after_prediction

    baseline_correct = kept + lost
    return {
        **effect,
        "retain_fidelity": kept / baseline_correct if baseline_correct else None,
        "transitions": {
            "kept_correct": kept,
            "lost_correct": lost,
            "gained_correct": gained,
        },
        "output_consistency": {
            "comparable_pairs": comparable,
            "consistent_rate": consistent / comparable if comparable else None,
        },
    }
