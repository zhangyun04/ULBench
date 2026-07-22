"""Shared runtime types for the method / model / probe plug-in interfaces.

These dataclasses are the vocabulary that :mod:`ulbench.methods`,
:mod:`ulbench.models`, and the future ``ulbench.runner`` exchange. They are
runtime objects, not release schemas — the frozen result-record contract lives
in ``docs/benchmark_spec_v1.md`` §6.5 and :mod:`ulbench.schema`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ulbench.schema import ResponseStatus


class CapabilityMismatchError(RuntimeError):
    """A method/model combination lacks required capabilities.

    Per spec §4.4 this must surface *before* model loading or intervention and
    must serialize to a manifest-level unsupported record — never crash after
    partial evaluation or be imputed as zero.
    """

    REASON_CODE = "CAPABILITY_MISMATCH"

    def __init__(self, missing_capabilities: list[str], context: str = ""):
        self.missing_capabilities = sorted(missing_capabilities)
        self.context = context
        detail = ", ".join(self.missing_capabilities)
        message = f"missing capabilities: {detail}"
        if context:
            message = f"{context}: {message}"
        super().__init__(message)

    def to_record(self) -> dict[str, Any]:
        """Return the spec §4.4 manifest-level unsupported record."""
        return {
            "status": "unsupported",
            "reason_code": self.REASON_CODE,
            "missing_capabilities": self.missing_capabilities,
        }


@dataclass
class ProbeRequest:
    """One evaluation request before/after method input transforms.

    ``method_instructions`` collects instruction lines injected by R0 methods;
    the probe renderer decides their placement in the final ``prompt_text`` so
    methods stay independent of prompt layout.
    """

    request_id: str
    item_id: str
    question: str
    choices: Optional[list[str]]
    image_path: Optional[str]
    prompt_text: Optional[str] = None
    method_instructions: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProbeResponse:
    """One model response with explicit status accounting.

    ``response_status`` is ``None`` while the response awaits scoring —
    adapters only assign operational statuses (``model_error``/``api_error``);
    ``correct``/``incorrect``/``refusal``/``invalid_format`` are the scorer's
    decision. Refusal, invalid format, and operational errors stay distinct
    and are never silently collapsed into ``incorrect`` (spec §6.5, §10.1).
    """

    request_id: str
    raw_output: str = ""
    prediction: Optional[str] = None
    response_status: Optional[ResponseStatus] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    extra: dict[str, Any] = field(default_factory=dict)
