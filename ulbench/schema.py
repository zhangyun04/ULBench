"""Normative ULBench v1 schemas and per-record validation.

This module intentionally uses the Python standard library instead of adding a
new runtime dependency. Cross-record invariants live in :mod:`ulbench.validation`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Optional, TypeVar

from vqa_gen.ontology.normalize import normalize_choice_text


SCHEMA_VERSION = "1.0.0"


class SchemaValidationError(ValueError):
    """One or more schema or cross-record validation failures."""

    def __init__(self, context: str, errors: list[str]):
        self.context = context
        self.errors = errors
        detail = "\n".join(f"- {error}" for error in errors)
        super().__init__(f"{context} validation failed:\n{detail}")


class AccessRegime(str, Enum):
    R0_BLACK_BOX = "R0_BLACK_BOX"
    R1_INFERENCE_STATE = "R1_INFERENCE_STATE"
    R2_WHITE_BOX = "R2_WHITE_BOX"


class ModelState(str, Enum):
    M0 = "M0"
    M_ACQ = "M_acq"
    M_U = "M_u"


class ProbeFamily(str, Enum):
    P0_CONTROL = "P0_CONTROL"
    P1_DIRECT = "P1_DIRECT"
    P2_INDIRECT = "P2_INDIRECT"
    P3_ADVERSARIAL = "P3_ADVERSARIAL"


class QuestionFormat(str, Enum):
    MCQ = "mcq"
    SHORT_ANSWER = "short_answer"
    MATCHING = "matching"


class InputCondition(str, Enum):
    NORMAL_IMAGE = "normal_image"
    NO_IMAGE = "no_image"
    SHUFFLED_IMAGE = "shuffled_image"
    OPTION_ONLY = "option_only"
    QUESTION_ONLY = "question_only"


class Split(str, Enum):
    TRAIN_FORGET = "train_forget"
    TRAIN_RETAIN = "train_retain"
    TEST_FORGET = "test_forget"
    TEST_RETAIN = "test_retain"
    UTILITY = "utility"
    AUDIT = "audit"


class ResponseStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    REFUSAL = "refusal"
    INVALID_FORMAT = "invalid_format"
    API_ERROR = "api_error"
    MODEL_ERROR = "model_error"
    POLICY_BLOCK = "policy_block"


class RunStatus(str, Enum):
    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


EnumT = TypeVar("EnumT", bound=Enum)


def _parse_enum(enum_type: type[EnumT], value: Any, field_name: str,
                errors: list[str]) -> Any:
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        errors.append(
            f"{field_name} must be one of {[member.value for member in enum_type]}; "
            f"got {value!r}"
        )
        return value


def _require_nonempty_string(value: Any, field_name: str, errors: list[str]):
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field_name} must be a non-empty string")


def _require_string_list(value: Any, field_name: str, errors: list[str],
                         *, nonempty: bool = False):
    if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
        errors.append(f"{field_name} must be a list of strings")
    elif nonempty and not value:
        errors.append(f"{field_name} must not be empty")


def _strict_dataclass_payload(cls, payload: dict[str, Any]) -> list[str]:
    if not isinstance(payload, dict):
        return ["payload must be an object"]
    expected = {field.name for field in fields(cls)}
    actual = set(payload)
    errors = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown fields: {', '.join(unknown)}")
    return errors


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    return value


@dataclass
class BenchmarkItem:
    schema_version: str
    item_id: str
    sample_id: str
    dataset_id: str
    image_id: str
    image: Optional[str]
    concept_id: str
    concept_name: str
    concept_aliases: list[str]
    forgetting_level: str
    concept_axis: str
    split: Split
    probe_id: str
    probe_family: ProbeFamily
    question_format: QuestionFormat
    prompt_variant_id: str
    input_condition: InputCondition
    question: str
    choices: Optional[list[str]]
    accepted_answers: list[str]
    answer_index: Optional[int]
    matched_retain_id: Optional[str]
    source: dict[str, Any]
    license: dict[str, Any]
    provenance_note: str
    metadata: dict[str, Any]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BenchmarkItem":
        errors = _strict_dataclass_payload(cls, payload)
        if errors:
            raise SchemaValidationError("BenchmarkItem", errors)
        values = dict(payload)
        values["split"] = _parse_enum(Split, values["split"], "split", errors)
        values["probe_family"] = _parse_enum(
            ProbeFamily, values["probe_family"], "probe_family", errors
        )
        values["question_format"] = _parse_enum(
            QuestionFormat, values["question_format"], "question_format", errors
        )
        values["input_condition"] = _parse_enum(
            InputCondition, values["input_condition"], "input_condition", errors
        )
        if errors:
            raise SchemaValidationError("BenchmarkItem", errors)
        item = cls(**values)
        item.validate()
        return item

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))

    def validate(self):
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )

        for field_name in (
            "item_id", "sample_id", "dataset_id", "image_id", "concept_id",
            "concept_name", "forgetting_level", "concept_axis", "probe_id",
            "prompt_variant_id", "provenance_note",
        ):
            _require_nonempty_string(getattr(self, field_name), field_name, errors)

        _require_string_list(self.concept_aliases, "concept_aliases", errors)
        _require_string_list(
            self.accepted_answers, "accepted_answers", errors, nonempty=True
        )

        if not isinstance(self.question, str):
            errors.append("question must be a string")
        elif self.input_condition == InputCondition.OPTION_ONLY:
            if self.question.strip():
                errors.append("question must be empty for option_only")
        elif not self.question.strip():
            errors.append("question must be non-empty outside option_only")

        if self.input_condition in {
            InputCondition.NO_IMAGE,
            InputCondition.OPTION_ONLY,
            InputCondition.QUESTION_ONLY,
        }:
            if self.image is not None:
                errors.append(f"image must be null for {self.input_condition.value}")
        elif not isinstance(self.image, str) or not self.image.strip():
            errors.append(f"image must be non-empty for {self.input_condition.value}")

        if self.input_condition == InputCondition.SHUFFLED_IMAGE:
            donor = self.metadata.get("donor_image_id") if isinstance(self.metadata, dict) else None
            if not isinstance(donor, str) or not donor:
                errors.append("metadata.donor_image_id is required for shuffled_image")
            elif donor == self.image_id:
                errors.append("shuffled_image donor_image_id must differ from image_id")

        if self.question_format == QuestionFormat.MCQ:
            if not isinstance(self.choices, list) or len(self.choices) < 2:
                errors.append("choices must contain at least two strings for mcq")
            elif any(not isinstance(choice, str) or not choice.strip()
                     for choice in self.choices):
                errors.append("every mcq choice must be a non-empty string")
            else:
                normalized_choices = [normalize_choice_text(choice) for choice in self.choices]
                if len(normalized_choices) != len(set(normalized_choices)):
                    errors.append("mcq choices must be unique after normalization")
                if not isinstance(self.answer_index, int) or not (
                    0 <= self.answer_index < len(self.choices)
                ):
                    errors.append("answer_index must be in range for mcq")
                elif normalize_choice_text(self.choices[self.answer_index]) not in {
                    normalize_choice_text(answer) for answer in self.accepted_answers
                }:
                    errors.append("accepted_answers must include the indexed mcq answer")
        elif self.question_format == QuestionFormat.SHORT_ANSWER:
            if self.choices is not None:
                errors.append(f"choices must be null for {self.question_format.value}")
            if self.answer_index is not None:
                errors.append(f"answer_index must be null for {self.question_format.value}")
        elif self.question_format == QuestionFormat.MATCHING:
            if self.choices is None:
                if self.answer_index is not None:
                    errors.append("answer_index must be null when matching has no choices")
            elif len(self.choices) < 2 or any(
                not isinstance(choice, str) or not choice.strip() for choice in self.choices
            ):
                errors.append("matching choices must contain at least two non-empty strings")
            elif not isinstance(self.answer_index, int) or not (
                0 <= self.answer_index < len(self.choices)
            ):
                errors.append("answer_index must be in range for choice-based matching")

        normalized_answers = [normalize_choice_text(answer)
                              for answer in self.accepted_answers
                              if isinstance(answer, str)]
        if len(normalized_answers) != len(set(normalized_answers)):
            errors.append("accepted_answers must be unique after normalization")

        if self.matched_retain_id is not None and (
            not isinstance(self.matched_retain_id, str) or not self.matched_retain_id.strip()
        ):
            errors.append("matched_retain_id must be null or a non-empty string")

        if not isinstance(self.source, dict):
            errors.append("source must be an object")
        else:
            for key in ("dataset_id", "record_id", "version"):
                _require_nonempty_string(self.source.get(key), f"source.{key}", errors)

        if not isinstance(self.license, dict):
            errors.append("license must be an object")
        else:
            for key in ("id", "redistribution"):
                _require_nonempty_string(self.license.get(key), f"license.{key}", errors)

        if not isinstance(self.metadata, dict):
            errors.append("metadata must be an object")
        else:
            _require_nonempty_string(
                self.metadata.get("pairing_id"), "metadata.pairing_id", errors
            )

        if errors:
            raise SchemaValidationError(f"BenchmarkItem[{self.item_id}]", errors)


@dataclass
class ProbeSpec:
    schema_version: str
    probe_id: str
    probe_version: str
    probe_family: ProbeFamily
    question_format: QuestionFormat
    input_condition: InputCondition
    allowed_modalities: list[str]
    prompt_template: str
    prompt_variant_id: str
    scorer_id: str
    scorer_version: str
    answer_space: dict[str, Any]
    construction_source: str
    seed: int
    applicable_concept_axes: list[str]
    reveals_concept_name: bool
    attack_family: Optional[str]
    attack_unit_cost: int
    expected_control_ids: list[str]
    known_limitations: list[str]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProbeSpec":
        errors = _strict_dataclass_payload(cls, payload)
        if errors:
            raise SchemaValidationError("ProbeSpec", errors)
        values = dict(payload)
        for name, enum_type in (
            ("probe_family", ProbeFamily),
            ("question_format", QuestionFormat),
            ("input_condition", InputCondition),
        ):
            values[name] = _parse_enum(enum_type, values[name], name, errors)
        if errors:
            raise SchemaValidationError("ProbeSpec", errors)
        spec = cls(**values)
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))

    def validate(self):
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
        for name in (
            "probe_id", "probe_version", "prompt_variant_id", "scorer_id",
            "scorer_version", "construction_source",
        ):
            _require_nonempty_string(getattr(self, name), name, errors)
        _require_string_list(self.allowed_modalities, "allowed_modalities", errors,
                             nonempty=True)
        _require_string_list(
            self.applicable_concept_axes, "applicable_concept_axes", errors,
            nonempty=True,
        )
        _require_string_list(self.expected_control_ids, "expected_control_ids", errors)
        _require_string_list(self.known_limitations, "known_limitations", errors)
        if not isinstance(self.answer_space, dict):
            errors.append("answer_space must be an object")
        if not isinstance(self.seed, int):
            errors.append("seed must be an integer")
        if not isinstance(self.reveals_concept_name, bool):
            errors.append("reveals_concept_name must be boolean")
        if not isinstance(self.attack_unit_cost, int) or self.attack_unit_cost < 0:
            errors.append("attack_unit_cost must be a non-negative integer")
        if self.probe_family == ProbeFamily.P3_ADVERSARIAL:
            _require_nonempty_string(self.attack_family, "attack_family", errors)
            if self.attack_unit_cost < 1:
                errors.append("P3 attack_unit_cost must be at least one")
        if errors:
            raise SchemaValidationError(f"ProbeSpec[{self.probe_id}]", errors)


@dataclass
class ModelCapabilities:
    supports_images: bool
    supports_logits: bool
    supports_hidden_states: bool
    supports_gradients: bool
    supports_weight_write: bool
    supports_system_prompt: bool
    supports_multi_turn: bool
    is_closed_api: bool
    constraints: dict[str, Any]

    CAPABILITY_FIELDS: ClassVar[tuple[str, ...]] = (
        "supports_images", "supports_logits", "supports_hidden_states",
        "supports_gradients", "supports_weight_write", "supports_system_prompt",
        "supports_multi_turn", "is_closed_api",
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelCapabilities":
        errors = _strict_dataclass_payload(cls, payload)
        if errors:
            raise SchemaValidationError("ModelCapabilities", errors)
        capabilities = cls(**payload)
        capabilities.validate()
        return capabilities

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self):
        errors = []
        for name in self.CAPABILITY_FIELDS:
            if not isinstance(getattr(self, name), bool):
                errors.append(f"{name} must be boolean")
        if not isinstance(self.constraints, dict):
            errors.append("constraints must be an object")
        if errors:
            raise SchemaValidationError("ModelCapabilities", errors)


@dataclass
class MethodSpec:
    schema_version: str
    method_id: str
    method_version: str
    access_regime: AccessRegime
    method_family: str
    semantic_label: str
    required_capabilities: list[str]
    requires_forget_set: bool
    requires_retain_set: bool
    uses_external_models: bool
    modifies_persistent_state: bool
    tunable_hyperparameters: dict[str, Any]
    selected_hyperparameters: dict[str, Any]
    tuning: dict[str, Any]
    cost_fields: list[str]
    metadata: dict[str, Any]

    SEMANTIC_LABELS: ClassVar[set[str]] = {
        "no_op", "behavioral_suppression", "output_filter",
        "inference_intervention", "weight_intervention",
    }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MethodSpec":
        errors = _strict_dataclass_payload(cls, payload)
        if errors:
            raise SchemaValidationError("MethodSpec", errors)
        values = dict(payload)
        values["access_regime"] = _parse_enum(
            AccessRegime, values["access_regime"], "access_regime", errors
        )
        if errors:
            raise SchemaValidationError("MethodSpec", errors)
        spec = cls(**values)
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))

    def validate(self):
        errors = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
        for name in ("method_id", "method_version", "method_family"):
            _require_nonempty_string(getattr(self, name), name, errors)
        if self.semantic_label not in self.SEMANTIC_LABELS:
            errors.append(f"semantic_label must be one of {sorted(self.SEMANTIC_LABELS)}")
        _require_string_list(
            self.required_capabilities, "required_capabilities", errors
        )
        unknown_capabilities = sorted(
            set(self.required_capabilities) - set(ModelCapabilities.CAPABILITY_FIELDS)
        )
        if unknown_capabilities:
            errors.append(
                f"unknown required_capabilities: {', '.join(unknown_capabilities)}"
            )
        for name in (
            "requires_forget_set", "requires_retain_set", "uses_external_models",
            "modifies_persistent_state",
        ):
            if not isinstance(getattr(self, name), bool):
                errors.append(f"{name} must be boolean")
        for name in (
            "tunable_hyperparameters", "selected_hyperparameters", "tuning", "metadata",
        ):
            if not isinstance(getattr(self, name), dict):
                errors.append(f"{name} must be an object")
        _require_string_list(self.cost_fields, "cost_fields", errors)
        if self.access_regime == AccessRegime.R2_WHITE_BOX and not self.modifies_persistent_state:
            errors.append("R2_WHITE_BOX methods must declare modifies_persistent_state=true")
        if self.semantic_label == "no_op" and self.modifies_persistent_state:
            errors.append("no_op cannot modify persistent state")
        if errors:
            raise SchemaValidationError(f"MethodSpec[{self.method_id}]", errors)


@dataclass
class RunManifest:
    schema_version: str
    run_id: str
    run_scope: str
    status: RunStatus
    created_at: str
    completed_at: Optional[str]
    repository: dict[str, Any]
    data: dict[str, Any]
    model: dict[str, Any]
    model_capabilities: ModelCapabilities
    model_state_input: ModelState
    model_state_output: ModelState
    method: dict[str, Any]
    access_regime: AccessRegime
    capability_validation: dict[str, Any]
    seeds: dict[str, int]
    inference: dict[str, Any]
    training: dict[str, Any]
    cost: dict[str, Any]
    environment: dict[str, Any]
    eligibility: dict[str, Any]
    artifacts: dict[str, Any]
    failure_counts: dict[str, int]

    RUN_SCOPES: ClassVar[set[str]] = {"main", "pilot", "audit", "legacy"}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RunManifest":
        errors = _strict_dataclass_payload(cls, payload)
        if errors:
            raise SchemaValidationError("RunManifest", errors)
        values = dict(payload)
        for name, enum_type in (
            ("status", RunStatus),
            ("model_state_input", ModelState),
            ("model_state_output", ModelState),
            ("access_regime", AccessRegime),
        ):
            values[name] = _parse_enum(enum_type, values[name], name, errors)
        try:
            values["model_capabilities"] = ModelCapabilities.from_dict(
                values["model_capabilities"]
            )
        except SchemaValidationError as exc:
            errors.extend(f"model_capabilities: {error}" for error in exc.errors)
        if errors:
            raise SchemaValidationError("RunManifest", errors)
        manifest = cls(**values)
        manifest.validate()
        return manifest

    def to_dict(self) -> dict[str, Any]:
        payload = _serialize(asdict(self))
        return payload

    @staticmethod
    def _require_keys(container: Any, name: str, required: set[str],
                      errors: list[str]):
        if not isinstance(container, dict):
            errors.append(f"{name} must be an object")
            return
        missing = sorted(required - set(container))
        if missing:
            errors.append(f"{name} missing keys: {', '.join(missing)}")

    def validate(self):
        errors = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
        _require_nonempty_string(self.run_id, "run_id", errors)
        if self.run_scope not in self.RUN_SCOPES:
            errors.append(f"run_scope must be one of {sorted(self.RUN_SCOPES)}")
        _require_nonempty_string(self.created_at, "created_at", errors)

        self._require_keys(
            self.repository, "repository", {"url", "git_sha", "dirty"}, errors
        )
        self._require_keys(
            self.data,
            "data",
            {
                "benchmark_version", "dataset_version", "dataset_hash", "split_hash",
                "concept_registry_hash", "probe_bank_hash", "prompt_bank_hash",
            },
            errors,
        )
        self._require_keys(
            self.model,
            "model",
            {
                "model_id", "model_revision", "processor_revision", "loader",
                "chat_template",
            },
            errors,
        )
        self._require_keys(
            self.method, "method", {"method_id", "method_version", "config_hash"},
            errors,
        )
        self._require_keys(
            self.capability_validation,
            "capability_validation",
            {"supported", "missing_capabilities"},
            errors,
        )
        self._require_keys(
            self.seeds,
            "seeds",
            {"data_selection", "option_order", "controls", "inference", "training"},
            errors,
        )
        self._require_keys(
            self.inference, "inference", {"decoding", "dtype", "device"}, errors
        )
        self._require_keys(
            self.training,
            "training",
            {"steps", "updated_parameter_count", "updated_parameter_fraction", "tuning_budget"},
            errors,
        )
        self._require_keys(
            self.cost,
            "cost",
            {"wall_seconds", "gpu_hours", "peak_memory_bytes", "inference_latency_ms"},
            errors,
        )
        for name in ("environment", "eligibility", "artifacts", "failure_counts"):
            if not isinstance(getattr(self, name), dict):
                errors.append(f"{name} must be an object")

        if self.run_scope == "main":
            self._require_keys(
                self.eligibility,
                "eligibility",
                {"manifest_id", "manifest_hash", "candidate_count", "eligible_count"},
                errors,
            )
        if self.status == RunStatus.UNSUPPORTED:
            if self.capability_validation.get("supported") is not False:
                errors.append("unsupported runs require capability_validation.supported=false")
            if not self.capability_validation.get("missing_capabilities"):
                errors.append("unsupported runs require missing_capabilities")
        if errors:
            raise SchemaValidationError(f"RunManifest[{self.run_id}]", errors)


def validate_method_model_compatibility(
    method: MethodSpec, capabilities: ModelCapabilities
) -> list[str]:
    """Return method-required capabilities unavailable on the model."""

    method.validate()
    capabilities.validate()
    return sorted(
        name for name in set(method.required_capabilities)
        if not getattr(capabilities, name)
    )
