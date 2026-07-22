"""Method-side plug-in contract (roadmap B-P0.2).

An :class:`UnlearningMethod` is an intervention the benchmark evaluates — it
is *not* part of the benchmark itself. The runner treats every method through
this interface only; adding a new method must never require runner changes.

Oracle prompts are instruction-following controls, not unlearning methods
(frozen decision #9 in HANDOUT.md); they live in
:mod:`ulbench.audits.oracle_controls` and deliberately do not subclass
:class:`UnlearningMethod` so they cannot enter method rankings.
"""

from __future__ import annotations

from typing import Any, Optional

from ulbench.schema import (
    MethodSpec,
    ModelCapabilities,
    validate_method_model_compatibility,
)
from ulbench.types import CapabilityMismatchError, ProbeRequest, ProbeResponse


class UnlearningMethod:
    """Base class for all evaluated interventions.

    Subclasses must provide ``spec`` (a validated :class:`MethodSpec`) and
    override only the hooks their access regime actually uses:

    - ``prepare``           one-time setup with forget/retain *training* data
    - ``transform_input``   R0 input-side change (prompt injection, ICL, ...)
    - ``intervene``         R1/R2 change to inference state or weights
    - ``transform_output``  R0 output-side change (filtering, rewriting)
    """

    spec: MethodSpec

    def __init__(self, spec: MethodSpec):
        spec.validate()
        self.spec = spec

    # ── capability contract (spec §4.3–4.4) ───────────────────────────
    def validate_against(self, capabilities: ModelCapabilities) -> None:
        """Raise :class:`CapabilityMismatchError` before any model loading."""
        missing = validate_method_model_compatibility(self.spec, capabilities)
        if missing:
            raise CapabilityMismatchError(
                missing,
                context=f"method {self.spec.method_id!r}",
            )

    # ── lifecycle hooks (all optional) ────────────────────────────────
    def prepare(self, model_adapter: Any, forget_set: Optional[list] = None,
                retain_set: Optional[list] = None,
                config: Optional[dict] = None) -> None:
        """One-time setup. Default: nothing."""

    def transform_input(self, request: ProbeRequest) -> ProbeRequest:
        """Input-side intervention. Default: identity."""
        return request

    def intervene(self, model_state: Any) -> Any:
        """Inference-state / weight intervention. Default: identity."""
        return model_state

    def transform_output(self, response: ProbeResponse) -> ProbeResponse:
        """Output-side intervention. Default: identity."""
        return response

    def metadata(self) -> dict[str, Any]:
        """Disclosure record for the run manifest."""
        return self.spec.to_dict()
