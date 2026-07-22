"""Model-side plug-in contract (roadmap B-P0.3).

A :class:`ModelAdapter` hides all model-family branching behind one interface
so the runner and evaluators contain no model special cases. Capabilities are
declared statically — before any weights load — because capability validation
must happen before model loading (spec §4.4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ulbench.schema import ModelCapabilities, ResponseStatus
from ulbench.types import CapabilityMismatchError, ProbeRequest, ProbeResponse

# Keys every adapter's model_info() must provide; mirrors the RunManifest
# ``model`` block so manifests can be filled without adapter-specific code.
MODEL_INFO_KEYS = (
    "model_id", "model_revision", "processor_revision", "loader", "chat_template",
)


class ModelAdapter(ABC):
    """Uniform inference interface over one model.

    Lifecycle: construct (cheap, no weights) → ``capabilities()`` /
    ``model_info()`` for validation and manifests → ``load()`` → inference via
    ``generate`` and, when supported, ``score_options``.
    """

    is_thinking: bool = False

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Static capability declaration; must not require loaded weights."""

    @abstractmethod
    def model_info(self) -> dict[str, Any]:
        """Manifest ``model`` block; must contain :data:`MODEL_INFO_KEYS`."""

    def load(self) -> None:
        """Load weights/processor. Idempotent. Default: nothing to load."""

    @abstractmethod
    def generate(self, requests: list[ProbeRequest]) -> list[ProbeResponse]:
        """Free-generation inference.

        Must return exactly one :class:`ProbeResponse` per request, aligned by
        ``request_id``, with failures reported as explicit statuses — never
        dropped or silently retried on a different path.
        """

    def score_options(self, requests: list[ProbeRequest],
                      option_count: int = 4) -> list[ProbeResponse]:
        """Logit-based option scoring. Only valid when ``supports_logits``."""
        raise CapabilityMismatchError(
            ["supports_logits"], context=f"adapter {type(self).__name__}"
        )

    def require_capability(self, name: str) -> None:
        if not getattr(self.capabilities(), name):
            raise CapabilityMismatchError(
                [name], context=f"adapter {type(self).__name__}"
            )


def check_adapter_contract(adapter: ModelAdapter,
                           requests: list[ProbeRequest]) -> list[str]:
    """10-item contract test helper (roadmap B-P0.3).

    Runs ``generate`` on *requests* and returns a list of violations —
    identical input fields, identical output schema, no silent fallback.
    An empty list means the adapter honors the contract.
    """
    violations: list[str] = []

    capabilities = adapter.capabilities()
    try:
        capabilities.validate()
    except Exception as exc:
        violations.append(f"capabilities invalid: {exc}")

    info = adapter.model_info()
    missing = [key for key in MODEL_INFO_KEYS if key not in info]
    if missing:
        violations.append(f"model_info missing keys: {', '.join(missing)}")

    responses = adapter.generate(requests)
    if len(responses) != len(requests):
        violations.append(
            f"generate returned {len(responses)} responses for "
            f"{len(requests)} requests"
        )
        return violations

    for request, response in zip(requests, responses):
        if not isinstance(response, ProbeResponse):
            violations.append(f"{request.request_id}: response is not a ProbeResponse")
            continue
        if response.request_id != request.request_id:
            violations.append(
                f"{request.request_id}: misaligned response {response.request_id}"
            )
        if response.response_status is not None and not isinstance(
            response.response_status, ResponseStatus
        ):
            violations.append(
                f"{request.request_id}: response_status must be a ResponseStatus or None"
            )
        if not isinstance(response.raw_output, str):
            violations.append(f"{request.request_id}: raw_output must be a string")
        if response.response_status in (
            ResponseStatus.MODEL_ERROR, ResponseStatus.API_ERROR,
        ) and not response.error:
            violations.append(
                f"{request.request_id}: error status without structured error detail"
            )
        if response.response_status is None and not response.raw_output:
            violations.append(
                f"{request.request_id}: empty output must carry an explicit "
                "error status, not pending-scoring"
            )
    return violations
