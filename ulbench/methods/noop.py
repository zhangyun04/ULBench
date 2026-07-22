"""No-op baseline: the untreated model (roadmap B-P0.7 R0 row)."""

from __future__ import annotations

from ulbench.methods.base import UnlearningMethod
from ulbench.schema import SCHEMA_VERSION, AccessRegime, MethodSpec


def noop_spec() -> MethodSpec:
    return MethodSpec.from_dict({
        "schema_version": SCHEMA_VERSION,
        "method_id": "no_op",
        "method_version": "1.0.0",
        "access_regime": AccessRegime.R0_BLACK_BOX.value,
        "method_family": "control",
        "semantic_label": "no_op",
        "required_capabilities": [],
        "requires_forget_set": False,
        "requires_retain_set": False,
        "uses_external_models": False,
        "modifies_persistent_state": False,
        "tunable_hyperparameters": {},
        "selected_hyperparameters": {},
        "tuning": {"procedure": "none"},
        "cost_fields": [],
        "metadata": {"legacy_condition": "BASELINE_NORMAL"},
    })


class NoOpMethod(UnlearningMethod):
    """Evaluates the model exactly as shipped. All hooks stay identity."""

    def __init__(self):
        super().__init__(noop_spec())
