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

import logging
import time
from typing import Any, Optional

from ulbench.schema import ModelCapabilities, ResponseStatus
from ulbench.types import ProbeRequest, ProbeResponse
from ulbench.models.base import ModelAdapter


logger = logging.getLogger(__name__)


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

    # ── text-only inference (P0 no_image / option_only controls) ─────
    # The legacy inference paths require an image; control conditions
    # deliberately omit it, so the adapter owns the text-only branch.

    def _text_only_batch(self, prompts: list[str]):
        """Return (inputs, device) for a text-only prompt batch."""
        import torch  # local: keep ulbench import light

        legacy = _legacy()
        texts = []
        for prompt in prompts:
            messages = [{"role": "user",
                         "content": [{"type": "text", "text": prompt}]}]
            texts.append(legacy._apply_chat_template(self.processor, messages))
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        device = legacy._get_first_device(self.model)
        return inputs.to(device), torch

    def _generate_text_only(self, prompts: list[str]) -> list[str]:
        try:
            inputs, torch = self._text_only_batch(prompts)
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=64, do_sample=False,
                )
            generated = output_ids[:, inputs.input_ids.shape[1]:]
            return self.processor.batch_decode(
                generated, skip_special_tokens=True,
            )
        except Exception as exc:
            logger.warning("Text-only generate failed: %s", exc)
            return [""] * len(prompts)

    def _score_text_only(self, prompts: list[str]) -> list[Optional[int]]:
        legacy = _legacy()
        try:
            inputs, torch = self._text_only_batch(prompts)
            with torch.no_grad():
                out = self.model(**inputs)
            last_logits = out.logits[:, -1, :]
            return [legacy._logit_pick(last_logits[index],
                                       self._answer_token_ids)
                    for index in range(len(prompts))]
        except Exception as exc:
            logger.warning("Text-only logit scoring failed: %s", exc)
            return [None] * len(prompts)

    @staticmethod
    def _partition_by_image(requests: list[ProbeRequest]):
        with_image, text_only = [], []
        for index, request in enumerate(requests):
            (with_image if request.image_path else text_only).append(index)
        return with_image, text_only

    def generate(self, requests: list[ProbeRequest]) -> list[ProbeResponse]:
        self.load()
        prompts = [request.prompt_text or "" for request in requests]
        start = time.time()

        with_image, text_only = self._partition_by_image(requests)
        raw_outputs: list[str] = [""] * len(requests)
        if with_image:
            outputs = _legacy().run_batch_inference(
                self.model, self.processor,
                [prompts[index] for index in with_image],
                [requests[index].image_path for index in with_image],
            )
            for index, output in zip(with_image, outputs):
                raw_outputs[index] = output
        if text_only:
            outputs = self._generate_text_only(
                [prompts[index] for index in text_only]
            )
            for index, output in zip(text_only, outputs):
                raw_outputs[index] = output

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
        start = time.time()

        with_image, text_only = self._partition_by_image(requests)
        picks: list[Optional[int]] = [None] * len(requests)
        if self.thinking_mode == "enabled":
            # Opt-in thinking-mode analysis: generate reasoning, then score.
            # Only defined for imaged requests; text-only stays unsupported.
            for index in with_image:
                try:
                    picks[index] = legacy.run_logit_thinking(
                        self.model, self.processor, prompts[index],
                        requests[index].image_path, self._answer_token_ids,
                    )
                except Exception:
                    picks[index] = None
        else:
            # Default: choice_logprob with thinking disabled at the template
            # level (legacy._apply_chat_template), for every checkpoint.
            if with_image:
                imaged_picks = legacy.run_logit_batch(
                    self.model, self.processor,
                    [prompts[index] for index in with_image],
                    [requests[index].image_path for index in with_image],
                    self._answer_token_ids,
                )
                for index, pick in zip(with_image, imaged_picks):
                    picks[index] = pick
            if text_only:
                text_picks = self._score_text_only(
                    [prompts[index] for index in text_only]
                )
                for index, pick in zip(text_only, text_picks):
                    picks[index] = pick
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
