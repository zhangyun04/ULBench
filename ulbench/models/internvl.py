"""InternVL adapter: native ``model.chat()`` inference path.

InternVL ships a bare tokenizer instead of a multimodal processor, so the
generic batched-forward logit path does not apply; generation goes through
the legacy ``run_internvl_batch`` chat wrapper and ``supports_logits`` is
declared false so the runner routes these models to generate+parse scoring.
"""

from __future__ import annotations

import time

from ulbench.schema import ModelCapabilities
from ulbench.types import ProbeRequest, ProbeResponse
from ulbench.models.huggingface import HuggingFaceAdapter, _legacy


class InternVLAdapter(HuggingFaceAdapter):

    loader = "internvl_chat"

    def capabilities(self) -> ModelCapabilities:
        capabilities = super().capabilities()
        capabilities.supports_logits = False
        capabilities.constraints["inference_path"] = "model.chat"
        return capabilities

    def generate(self, requests: list[ProbeRequest]) -> list[ProbeResponse]:
        self.load()
        prompts = [request.prompt_text or "" for request in requests]
        images = [request.image_path for request in requests]
        start = time.time()
        raw_outputs = _legacy().run_internvl_batch(
            self.model, self.processor, prompts, images
        )
        latency_ms = (time.time() - start) * 1000.0 / max(len(requests), 1)
        return self._responses_from_raw(requests, raw_outputs, latency_ms)
