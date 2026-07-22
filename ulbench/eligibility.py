"""Model-specific concept eligibility (spec §9, roadmap B-P0.5).

Consumes scored `M0` baseline records and produces the eligibility manifest
``eligible_concepts/<model_revision>.json``. Every candidate concept stays in
the manifest with machine-readable failure codes, so coverage is observable
and selection cannot be hidden.

Thresholds default to the preregistered pilot candidates from spec §9.3; any
change after G0 must go through the Decision Log before the full run.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from ulbench.ids import slug
from ulbench.metrics.statistics import bootstrap_ci, mean_or_none
from ulbench.schema import (
    SCHEMA_VERSION,
    InputCondition,
    QuestionFormat,
    ResponseStatus,
    SchemaValidationError,
)

# Spec §9.4 recommended failure codes.
INSUFFICIENT_SAMPLES = "INSUFFICIENT_SAMPLES"
LOW_NORMAL_ACCESS = "LOW_NORMAL_ACCESS"
LOW_VISUAL_GAP = "LOW_VISUAL_GAP"
CI_BELOW_FLOOR = "CI_BELOW_FLOOR"
VARIANT_INSTABILITY = "VARIANT_INSTABILITY"
MISSING_CONTROL = "MISSING_CONTROL"

_CONTROL_CONDITIONS = (
    InputCondition.NO_IMAGE,
    InputCondition.OPTION_ONLY,
    InputCondition.SHUFFLED_IMAGE,
)

# §9.4: manifest must carry these context entries.
REQUIRED_CONTEXT_KEYS = (
    "model_id", "model_revision", "processor_revision",
    "benchmark_version", "probe_bank_hash", "prompt_bank_hash",
    "scorer_versions", "data_hashes", "code_git_sha",
)


@dataclass(frozen=True)
class EligibilityThresholds:
    """Preregistered pilot candidate thresholds (spec §9.3)."""

    version: str = "pilot_candidate_v1"
    n_min: int = 50
    tau_acc_mcq: float = 0.60
    tau_acc_short: float = 0.50
    tau_visual: float = 0.20
    variant_min_accuracy: float = 0.50
    variant_max_spread: float = 0.15
    bootstrap_samples: int = 2000
    bootstrap_alpha: float = 0.05


@dataclass(frozen=True)
class BaselineRecord:
    """One scored M0 request used as eligibility evidence."""

    concept_id: str
    question_format: QuestionFormat
    input_condition: InputCondition
    prompt_variant_id: str
    response_status: ResponseStatus
    choice_count: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineRecord":
        errors = []
        try:
            return cls(
                concept_id=payload["concept_id"],
                question_format=QuestionFormat(payload["question_format"]),
                input_condition=InputCondition(payload["input_condition"]),
                prompt_variant_id=payload["prompt_variant_id"],
                response_status=ResponseStatus(payload["response_status"]),
                choice_count=payload.get("choice_count"),
            )
        except KeyError as exc:
            errors.append(f"missing field {exc.args[0]}")
        except ValueError as exc:
            errors.append(str(exc))
        raise SchemaValidationError("BaselineRecord", errors)

    @property
    def access(self) -> int:
        return 1 if self.response_status == ResponseStatus.CORRECT else 0


def _access_scores(records: list[BaselineRecord],
                   condition: InputCondition) -> list[int]:
    return [record.access for record in records
            if record.input_condition == condition]


def _evaluate_format(
    records: list[BaselineRecord],
    question_format: QuestionFormat,
    thresholds: EligibilityThresholds,
    seed: int,
) -> tuple[dict[str, Any], list[str]]:
    """Evaluate one concept × format cell; returns (stats, failure_codes)."""
    failures: list[str] = []
    format_records = [record for record in records
                      if record.question_format == question_format]
    normal = _access_scores(format_records, InputCondition.NORMAL_IMAGE)

    stats: dict[str, Any] = {
        "n_normal": len(normal),
        "access": {},
        "visual_gap": None,
        "ci": None,
        "chance_floor": None,
        "variants": {},
    }

    if len(normal) < thresholds.n_min:
        failures.append(INSUFFICIENT_SAMPLES)
    if not normal:
        return stats, failures

    normal_access = sum(normal) / len(normal)
    stats["access"]["normal_image"] = normal_access

    tau_acc = (thresholds.tau_acc_mcq
               if question_format == QuestionFormat.MCQ
               else thresholds.tau_acc_short)
    if normal_access < tau_acc:
        failures.append(LOW_NORMAL_ACCESS)

    control_access: list[float] = []
    missing_control = False
    for condition in _CONTROL_CONDITIONS:
        scores = _access_scores(format_records, condition)
        if not scores:
            missing_control = True
            stats["access"][condition.value] = None
            continue
        value = sum(scores) / len(scores)
        stats["access"][condition.value] = value
        control_access.append(value)
    if missing_control:
        failures.append(MISSING_CONTROL)
    else:
        stats["visual_gap"] = normal_access - max(control_access)
        if stats["visual_gap"] < thresholds.tau_visual:
            failures.append(LOW_VISUAL_GAP)

    if question_format == QuestionFormat.MCQ:
        choice_counts = {record.choice_count for record in format_records
                         if record.choice_count is not None}
        if len(choice_counts) != 1:
            raise SchemaValidationError(
                "eligibility",
                [f"mcq records need exactly one choice_count; got {sorted(choice_counts)}"],
            )
        chance_floor = 1.0 / next(iter(choice_counts))
        stats["chance_floor"] = chance_floor
        ci = bootstrap_ci(
            normal,
            num_samples=thresholds.bootstrap_samples,
            alpha=thresholds.bootstrap_alpha,
            seed=seed,
        )
        stats["ci"] = ci
        if ci["low"] <= chance_floor:
            failures.append(CI_BELOW_FLOOR)

        variant_access: dict[str, list[int]] = defaultdict(list)
        for record in format_records:
            if record.input_condition == InputCondition.NORMAL_IMAGE:
                variant_access[record.prompt_variant_id].append(record.access)
        per_variant = {
            variant: sum(scores) / len(scores)
            for variant, scores in sorted(variant_access.items())
        }
        stats["variants"] = per_variant
        if per_variant:
            values = list(per_variant.values())
            if (min(values) < thresholds.variant_min_accuracy
                    or max(values) - min(values) > thresholds.variant_max_spread):
                failures.append(VARIANT_INSTABILITY)

    return stats, failures


def build_eligibility_manifest(
    records: Iterable[dict[str, Any] | BaselineRecord],
    *,
    thresholds: Optional[EligibilityThresholds] = None,
    context: dict[str, Any],
    seed: int,
    required_formats: tuple[QuestionFormat, ...] = (
        QuestionFormat.MCQ, QuestionFormat.SHORT_ANSWER,
    ),
) -> dict[str, Any]:
    """Aggregate scored M0 records into the spec §9.4 manifest structure."""
    thresholds = thresholds or EligibilityThresholds()

    missing_context = sorted(set(REQUIRED_CONTEXT_KEYS) - set(context))
    if missing_context:
        raise SchemaValidationError(
            "eligibility context",
            [f"missing required context keys: {', '.join(missing_context)}"],
        )

    parsed: list[BaselineRecord] = []
    for record in records:
        if isinstance(record, BaselineRecord):
            parsed.append(record)
        else:
            parsed.append(BaselineRecord.from_dict(record))
    if not parsed:
        raise SchemaValidationError("eligibility", ["no baseline records provided"])

    by_concept: dict[str, list[BaselineRecord]] = defaultdict(list)
    for record in parsed:
        by_concept[record.concept_id].append(record)

    concepts: dict[str, Any] = {}
    eligible_count = 0
    for index, concept_id in enumerate(sorted(by_concept)):
        concept_records = by_concept[concept_id]
        failure_codes: list[str] = []
        formats: dict[str, Any] = {}
        for question_format in required_formats:
            stats, failures = _evaluate_format(
                concept_records, question_format, thresholds,
                seed=seed + index,
            )
            formats[question_format.value] = stats
            failure_codes.extend(
                f"{question_format.value}:{code}" for code in failures
            )
        eligible = not failure_codes
        eligible_count += eligible
        concepts[concept_id] = {
            "eligible": eligible,
            "failure_codes": failure_codes,
            "formats": formats,
        }

    candidate_count = len(concepts)
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "eligibility_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context": dict(context),
        "thresholds": asdict(thresholds),
        "seed": seed,
        "required_formats": [fmt.value for fmt in required_formats],
        "candidate_count": candidate_count,
        "eligible_count": eligible_count,
        "coverage": eligible_count / candidate_count,
        "concepts": concepts,
    }


def write_eligibility_manifest(manifest: dict[str, Any],
                               out_dir: str | Path) -> Path:
    """Write to ``<out_dir>/eligible_concepts/<model_revision>.json``."""
    revision = manifest["context"]["model_revision"]
    path = Path(out_dir) / "eligible_concepts" / f"{slug(str(revision))}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def summarize_coverage(manifest: dict[str, Any]) -> dict[str, Any]:
    """Coverage view for reporting: pass/fail counts per failure code."""
    code_counts: dict[str, int] = defaultdict(int)
    for concept in manifest["concepts"].values():
        for code in concept["failure_codes"]:
            code_counts[code] += 1
    return {
        "candidate_count": manifest["candidate_count"],
        "eligible_count": manifest["eligible_count"],
        "coverage": manifest["coverage"],
        "failure_code_counts": dict(sorted(code_counts.items())),
    }
