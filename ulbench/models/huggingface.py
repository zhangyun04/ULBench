"""HuggingFace open-weight VLM adapter.

Thin delegation layer over the battle-tested loading and inference code in
``experiments/intext_unlearning.py``. The legacy module stays the single
source of truth for GPU behavior while pipelines still run through it; when
``ulbench.runner`` becomes the primary entry point, the implementation moves
here and the legacy script becomes the wrapper (roadmap B-P0.2 compat
strategy). Legacy imports happen lazily so importing :mod:`ulbench` never
pulls in torch.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from ulbench.schema import ModelCapabilities, ResponseStatus
from ulbench.types import ProbeRequest, ProbeResponse
from ulbench.models.base import ModelAdapter


def _legacy():
    from experiments import intext_unlearning
    return intext_unlearning


class HuggingFaceAdapter(ModelAdapter):
    """Open-weight VLM served through transformers auto-classes."""

    loader = "hf_auto"

    def __init__(self, model_id: str, model_revision: Optional[str] = None,
                 gpu_ids: Optional[list[int]] = None,
                 thinking_mode: str = "disabled"):
        if thinking_mode not in ("disabled", "enabled"):
            raise ValueError(
                f"thinking_mode must be 'disabled' or 'enabled'; got {thinking_mode!r}"
            )
        self.model_id = model_id
        self.model_revision = model_revision
        self.gpu_ids = gpu_ids
        # Main experiments keep thinking disabled: *-Thinking checkpoints are
        # scored via choice_logprob with thinking off (see the runner decision
        # log). "enabled" is an opt-in for future thinking-mode analysis only.
        self.thinking_mode = thinking_mode
        self.model = None
        self.processor = None
        self._answer_token_ids = None

    @property
    def is_thinking_checkpoint(self) -> bool:
        """True when the checkpoint is a thinking model (for labeling only)."""
        return _legacy().is_thinking_model(self.model_id)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supports_images=True,
            supports_logits=True,
            supports_hidden_states=True,
            supports_gradients=True,
            supports_weight_write=True,
            supports_system_prompt=True,
            supports_multi_turn=True,
            is_closed_api=False,
            constraints={
                "declared_before_load": True,
                "revision_pinned": self.model_revision is not None,
            },
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision or "unpinned",
            "processor_revision": self.model_revision or "unpinned",
            "loader": self.loader,
            "chat_template": "processor_default",
        }

    def load(self) -> None:
        if self.model is not None:
            return
        self.model, self.processor = _legacy().load_model_and_processor(
            self.model_id, gpu_ids=self.gpu_ids
        )

    def _responses_from_raw(self, requests: list[ProbeRequest],
                            raw_outputs: list[str],
                            latency_ms: float) -> list[ProbeResponse]:
        responses = []
        for request, raw in zip(requests, raw_outputs):
            if raw:
                responses.append(ProbeResponse(
                    request_id=request.request_id,
                    raw_output=raw,
                    latency_ms=latency_ms,
                ))
            else:
                # Legacy inference signals per-item failure with "".
                responses.append(ProbeResponse(
                    request_id=request.request_id,
                    raw_output="",
                    response_status=ResponseStatus.MODEL_ERROR,
                    error="legacy_empty_output",
                    latency_ms=latency_ms,
                ))
        return responses

    def generate(self, requests: list[ProbeRequest]) -> list[ProbeResponse]:
        self.load()
        prompts = [request.prompt_text or "" for request in requests]
        images = [request.image_path for request in requests]
        start = time.time()
        raw_outputs = _legacy().run_batch_inference(
            self.model, self.processor, prompts, images
        )
        latency_ms = (time.time() - start) * 1000.0 / max(len(requests), 1)
        return self._responses_from_raw(requests, raw_outputs, latency_ms)

    def score_options(self, requests: list[ProbeRequest],
                      option_count: int = 4) -> list[ProbeResponse]:
        self.require_capability("supports_logits")
        self.load()
        legacy = _legacy()
        if self._answer_token_ids is None:
            self._answer_token_ids = legacy.get_answer_token_ids(self.processor)

        prompts = [request.prompt_text or "" for request in requests]
        images = [request.image_path for request in requests]
        start = time.time()
        if self.thinking_mode == "enabled":
            # Opt-in thinking-mode analysis: generate reasoning, then score.
            picks: list[Optional[int]] = []
            for prompt, image in zip(prompts, images):
                try:
                    picks.append(legacy.run_logit_thinking(
                        self.model, self.processor, prompt, image,
                        self._answer_token_ids,
                    ))
                except Exception:
                    picks.append(None)
        else:
            # Default: choice_logprob with thinking disabled at the template
            # level (legacy._apply_chat_template), for every checkpoint.
            picks = legacy.run_logit_batch(
                self.model, self.processor, prompts, images,
                self._answer_token_ids,
            )
        latency_ms = (time.time() - start) * 1000.0 / max(len(requests), 1)

        responses = []
        for request, pick in zip(requests, picks):
            if pick is None:
                responses.append(ProbeResponse(
                    request_id=request.request_id,
                    response_status=ResponseStatus.MODEL_ERROR,
                    error="logit_scoring_failed",
                    latency_ms=latency_ms,
                    extra={"scorer_id": "logit_argmax"},
                ))
            else:
                responses.append(ProbeResponse(
                    request_id=request.request_id,
                    raw_output=str(pick),
                    prediction=str(pick),
                    latency_ms=latency_ms,
                    extra={"scorer_id": "logit_argmax"},
                ))
        return responses
