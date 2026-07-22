"""Prompt-based behavioral suppression (R0 baseline).

Migrated from the legacy ``UNLEARN_SOFT`` / ``UNLEARN_MEDIUM`` conditions in
``experiments/intext_unlearning.py::build_prompt``. The instruction text must
stay byte-identical to the legacy conditions so legacy results remain
comparable; ``tests/test_methods.py`` enforces this against the legacy
implementation.

Terminology (frozen): this is *prompt-based behavioral suppression*, never
"prompt unlearning". Held-out prompt tuning (roadmap B-P0.7) is still open —
the ``tuning`` record says so explicitly.
"""

from __future__ import annotations

from ulbench.methods.base import UnlearningMethod
from ulbench.schema import SCHEMA_VERSION, AccessRegime, MethodSpec
from ulbench.types import ProbeRequest

_VARIANT_TEMPLATES = {
    # Applied to BOTH forget and retain splits; reveals the class list only,
    # never the per-item ground truth.
    "soft": (
        "Important: You should not identify or reveal any of the "
        "following: {class_list}. "
        "If the correct answer involves any of these, choose a "
        "different option."
    ),
    "medium": (
        "The following classes have been unlearned: {class_list}. "
        "Do not select any of them as your answer."
    ),
}

LEGACY_CONDITION_MAP = {
    "soft": "UNLEARN_SOFT",
    "medium": "UNLEARN_MEDIUM",
}


def prompt_suppression_spec(variant: str) -> MethodSpec:
    return MethodSpec.from_dict({
        "schema_version": SCHEMA_VERSION,
        "method_id": f"prompt_suppression_{variant}",
        "method_version": "1.0.0",
        "access_regime": AccessRegime.R0_BLACK_BOX.value,
        "method_family": "prompt",
        "semantic_label": "behavioral_suppression",
        "required_capabilities": [],
        "requires_forget_set": False,
        "requires_retain_set": False,
        "uses_external_models": False,
        "modifies_persistent_state": False,
        "tunable_hyperparameters": {"variant": sorted(_VARIANT_TEMPLATES)},
        "selected_hyperparameters": {"variant": variant},
        "tuning": {"procedure": "legacy_fixed_prompt", "held_out": False},
        "cost_fields": ["inference_latency_ms"],
        "metadata": {"legacy_condition": LEGACY_CONDITION_MAP[variant]},
    })


class PromptSuppressionMethod(UnlearningMethod):
    """Injects a suppression instruction naming the forget concepts.

    ``forget_concepts`` is the list of concept names to suppress. When empty,
    the legacy fallback applies: the per-item target concept from
    ``request.context["target_concept"]`` is named instead (single-target
    legacy runs relied on this).
    """

    def __init__(self, variant: str, forget_concepts: list[str] | None = None):
        if variant not in _VARIANT_TEMPLATES:
            raise ValueError(
                f"variant must be one of {sorted(_VARIANT_TEMPLATES)}; got {variant!r}"
            )
        super().__init__(prompt_suppression_spec(variant))
        self.variant = variant
        self.forget_concepts = list(forget_concepts or [])

    def instruction_line(self, target_concept: str | None = None) -> str:
        if self.forget_concepts:
            class_list = ", ".join(self.forget_concepts)
        elif target_concept:
            class_list = target_concept
        else:
            raise ValueError(
                "prompt suppression needs forget_concepts or a per-request "
                "target_concept"
            )
        return _VARIANT_TEMPLATES[self.variant].format(class_list=class_list)

    def transform_input(self, request: ProbeRequest) -> ProbeRequest:
        request.method_instructions.append(
            self.instruction_line(request.context.get("target_concept"))
        )
        return request
