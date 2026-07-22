"""Method-independent evaluation runner (roadmap B-P0.2, spec §5.2).

The runner accepts a model adapter, an unlearning-method plug-in, benchmark
items, and an output directory; it owns capability gating, prompt rendering,
method hooks, scoring, and the frozen artifact contract:

```
run_dir/
  run_manifest.json
  results.jsonl
  metrics.json
  logs/
  failures.jsonl
```

Adding a method or model must never require changes here — only the plug-in
interfaces in :mod:`ulbench.methods.base` and :mod:`ulbench.models.base`.
Unsupported method–model combinations produce a manifest-level record and no
metric files (spec §4.4); they never crash after partial evaluation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ulbench.metrics.accounting import response_accounting
from ulbench.methods.base import UnlearningMethod
from ulbench.models.base import ModelAdapter
from ulbench.probes.render import RENDERER_VERSION, render_prompt
from ulbench.probes.scorers import score_mcq, score_short_answer
from ulbench.schema import (
    SCHEMA_VERSION,
    BenchmarkItem,
    ModelState,
    QuestionFormat,
    ResponseStatus,
    RunManifest,
    RunStatus,
)
from ulbench.types import CapabilityMismatchError, ProbeRequest

logger = logging.getLogger(__name__)

_SCORABLE = {ResponseStatus.CORRECT, ResponseStatus.INCORRECT}


@dataclass
class RunConfig:
    run_id: str
    out_dir: str
    image_root: str
    run_scope: str = "pilot"
    model_state: ModelState = ModelState.M0
    batch_size: int = 8
    seed: int = 0
    max_new_tokens: int = 64
    dtype: str = "bfloat16"
    seeds: dict[str, int] = field(default_factory=dict)
    notes: dict[str, Any] = field(default_factory=dict)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_info() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    def run(*argv):
        try:
            return subprocess.run(
                ["git", *argv], cwd=root, capture_output=True, text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return ""
    sha = run("rev-parse", "HEAD") or "unknown"
    dirty = bool(run("status", "--porcelain"))
    url = run("remote", "get-url", "origin") or "unknown"
    return {"url": url, "git_sha": sha, "dirty": dirty}


def _environment_info() -> dict[str, Any]:
    import platform
    info = {"python": platform.python_version()}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def _peak_memory_bytes() -> Optional[int]:
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return None


def _data_block(items: list[BenchmarkItem]) -> dict[str, str]:
    canonical = json.dumps(
        [item.to_dict() for item in
         sorted(items, key=lambda item: item.item_id)],
        ensure_ascii=False, sort_keys=True,
    )
    return {
        "benchmark_version": SCHEMA_VERSION,
        "dataset_version": ",".join(sorted(
            {str(item.source.get("version", "unknown")) for item in items}
        )),
        "dataset_hash": _sha256_text(canonical),
        "split_hash": _sha256_text(json.dumps(sorted(
            (item.item_id, item.split.value) for item in items
        ))),
        "concept_registry_hash": _sha256_text(json.dumps(sorted(
            {item.concept_id for item in items}
        ))),
        "probe_bank_hash": _sha256_text(json.dumps(sorted(
            {item.probe_id for item in items}
        ))),
        "prompt_bank_hash": _sha256_text(json.dumps(sorted(
            {item.prompt_variant_id for item in items}
        ) + [RENDERER_VERSION])),
    }


class EvaluationRunner:
    """Runs one ``method × model × item set`` evaluation to frozen artifacts."""

    def __init__(self, adapter: ModelAdapter, method: UnlearningMethod,
                 config: RunConfig):
        self.adapter = adapter
        self.method = method
        self.config = config
        self.out_dir = Path(config.out_dir)

    # ── manifest ──────────────────────────────────────────────────────
    def _base_manifest(self, items: list[BenchmarkItem],
                       status: RunStatus,
                       capability_validation: dict[str, Any]) -> dict[str, Any]:
        config = self.config
        gpu_count = len(getattr(self.adapter, "gpu_ids", None) or [])
        seeds = {
            "data_selection": config.seed, "option_order": config.seed,
            "controls": config.seed, "inference": config.seed, "training": 0,
        }
        seeds.update(config.seeds)
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": config.run_id,
            "run_scope": config.run_scope,
            "status": status.value,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
            "repository": _git_info(),
            "data": _data_block(items),
            "model": self.adapter.model_info(),
            "model_capabilities": self.adapter.capabilities().to_dict(),
            "model_state_input": config.model_state.value,
            "model_state_output": config.model_state.value,
            "method": {
                "method_id": self.method.spec.method_id,
                "method_version": self.method.spec.method_version,
                "config_hash": _sha256_text(json.dumps(
                    self.method.metadata(), ensure_ascii=False, sort_keys=True
                )),
            },
            "access_regime": self.method.spec.access_regime.value,
            "capability_validation": capability_validation,
            "seeds": seeds,
            "inference": {
                "decoding": {"do_sample": False,
                             "max_new_tokens": config.max_new_tokens},
                "dtype": config.dtype,
                "device": f"cuda x{gpu_count}" if gpu_count else "auto",
                "batch_size": config.batch_size,
            },
            "training": {
                "steps": 0, "updated_parameter_count": 0,
                "updated_parameter_fraction": 0.0, "tuning_budget": "none",
            },
            "cost": {
                "wall_seconds": None, "gpu_hours": None,
                "peak_memory_bytes": None, "inference_latency_ms": None,
            },
            "environment": _environment_info(),
            "eligibility": {},
            "artifacts": {},
            "failure_counts": {},
        }

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        RunManifest.from_dict(payload)  # fail loudly on contract violations
        path = self.out_dir / "run_manifest.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    # ── request/response plumbing ─────────────────────────────────────
    def _build_request(self, item: BenchmarkItem) -> ProbeRequest:
        image_path = None
        if item.image is not None:
            image_path = os.path.join(self.config.image_root, item.image)
        request = ProbeRequest(
            request_id=f"{self.config.run_id}/{item.item_id}",
            item_id=item.item_id,
            question=item.question,
            choices=item.choices,
            image_path=image_path,
            context={
                "target_concept": item.concept_name,
                "split": item.split.value,
                "answer_index": item.answer_index,
            },
        )
        request = self.method.transform_input(request)
        request.prompt_text = render_prompt(item, request.method_instructions)
        return request

    def _record(self, item: BenchmarkItem, request: ProbeRequest,
                *, response_status: ResponseStatus, raw_output: str,
                prediction: Optional[str], scorer_id: str,
                scorer_version: str, latency_ms: Optional[float],
                error: Optional[str], detail: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": request.request_id,
            "run_id": self.config.run_id,
            "item_id": item.item_id,
            "sample_id": item.sample_id,
            "concept_id": item.concept_id,
            "split": item.split.value,
            "matched_retain_id": item.matched_retain_id,
            "model_state": self.config.model_state.value,
            "method_id": self.method.spec.method_id,
            "probe_id": item.probe_id,
            "probe_family": item.probe_family.value,
            "question_format": item.question_format.value,
            "prompt_variant_id": item.prompt_variant_id,
            "input_condition": item.input_condition.value,
            "answer_index": item.answer_index,
            "accepted_answers": item.accepted_answers,
            "option_order_hash": item.metadata.get("option_order_hash"),
            "raw_output": raw_output,
            "prediction": prediction,
            "scorer_id": scorer_id,
            "scorer_version": scorer_version,
            "response_status": response_status.value,
            "error": error,
            "latency_ms": latency_ms,
            "scorer_detail": detail,
        }

    def _run_batch(self, batch: list[tuple[BenchmarkItem, ProbeRequest]],
                   use_logits: bool) -> list[dict[str, Any]]:
        requests = [request for _, request in batch]
        if use_logits:
            responses = self.adapter.score_options(requests)
        else:
            responses = self.adapter.generate(requests)
        responses = [self.method.transform_output(response)
                     for response in responses]

        records = []
        for (item, request), response in zip(batch, responses):
            if response.response_status in (
                ResponseStatus.MODEL_ERROR, ResponseStatus.API_ERROR,
            ):
                records.append(self._record(
                    item, request,
                    response_status=response.response_status,
                    raw_output=response.raw_output or "",
                    prediction=None,
                    scorer_id="none", scorer_version="none",
                    latency_ms=response.latency_ms,
                    error=response.error or "unspecified adapter error",
                    detail={},
                ))
                continue

            if use_logits:
                if response.prediction is None:
                    records.append(self._record(
                        item, request,
                        response_status=ResponseStatus.MODEL_ERROR,
                        raw_output=response.raw_output or "",
                        prediction=None,
                        scorer_id="none", scorer_version="none",
                        latency_ms=response.latency_ms,
                        error="adapter returned no prediction",
                        detail={},
                    ))
                    continue
                predicted = int(response.prediction)
                status = (ResponseStatus.CORRECT
                          if predicted == item.answer_index
                          else ResponseStatus.INCORRECT)
                records.append(self._record(
                    item, request,
                    response_status=status,
                    raw_output=response.raw_output,
                    prediction=response.prediction,
                    scorer_id="logit_argmax", scorer_version="1.0.0",
                    latency_ms=response.latency_ms, error=None, detail={},
                ))
            else:
                if item.question_format == QuestionFormat.MCQ:
                    score = score_mcq(response.raw_output, item)
                else:
                    score = score_short_answer(response.raw_output, item)
                records.append(self._record(
                    item, request,
                    response_status=score.response_status,
                    raw_output=response.raw_output,
                    prediction=score.prediction,
                    scorer_id=score.scorer_id,
                    scorer_version=score.scorer_version,
                    latency_ms=response.latency_ms, error=response.error,
                    detail=score.detail,
                ))
        return records

    # ── metrics ───────────────────────────────────────────────────────
    def _metrics(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        def grouped(keys: tuple[str, ...]) -> dict[str, Any]:
            groups: dict[str, list[str]] = {}
            for record in records:
                label = "|".join(str(record[key]) for key in keys)
                groups.setdefault(label, []).append(record["response_status"])
            return {label: response_accounting(statuses)
                    for label, statuses in sorted(groups.items())}

        return {
            "run_id": self.config.run_id,
            "method_id": self.method.spec.method_id,
            "model_id": self.adapter.model_info()["model_id"],
            "model_state": self.config.model_state.value,
            "overall": response_accounting(
                record["response_status"] for record in records
            ),
            "by_condition": grouped(
                ("split", "question_format", "input_condition")
            ),
            "by_concept": grouped(("concept_id", "input_condition")),
        }

    # ── entry point ───────────────────────────────────────────────────
    def run(self, items: list[BenchmarkItem]) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = self.out_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        log_handler = logging.FileHandler(logs_dir / "runner.log")
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        )
        logger.addHandler(log_handler)
        try:
            return self._run(items)
        finally:
            logger.removeHandler(log_handler)
            log_handler.close()

    def _run(self, items: list[BenchmarkItem]) -> dict[str, Any]:
        config = self.config
        for item in items:
            item.validate()

        # Capability gate BEFORE any model loading (spec §4.4).
        try:
            self.method.validate_against(self.adapter.capabilities())
        except CapabilityMismatchError as exc:
            logger.warning("Unsupported combination: %s", exc)
            manifest = self._base_manifest(
                items, RunStatus.UNSUPPORTED,
                {"supported": False, **exc.to_record()},
            )
            self._write_manifest(manifest)
            return manifest

        manifest = self._base_manifest(
            items, RunStatus.RUNNING,
            {"supported": True, "missing_capabilities": []},
        )
        self._write_manifest(manifest)

        t_start = time.time()
        self.adapter.load()
        self.method.prepare(self.adapter)

        supports_logits = self.adapter.capabilities().supports_logits
        records: list[dict[str, Any]] = []
        pending: list[tuple[bool, list[tuple[BenchmarkItem, ProbeRequest]]]] = []

        # Preserve item order inside each scoring path; chunk to batch_size.
        mcq_items = [item for item in items
                     if item.question_format == QuestionFormat.MCQ]
        other_items = [item for item in items
                       if item.question_format != QuestionFormat.MCQ]
        for subset, use_logits in (
            (mcq_items, supports_logits), (other_items, False),
        ):
            batch: list[tuple[BenchmarkItem, ProbeRequest]] = []
            for item in subset:
                batch.append((item, self._build_request(item)))
                if len(batch) >= config.batch_size:
                    pending.append((use_logits, batch))
                    batch = []
            if batch:
                pending.append((use_logits, batch))

        for index, (use_logits, batch) in enumerate(pending):
            records.extend(self._run_batch(batch, use_logits))
            logger.info("run %s: batch %d/%d done (%d records)",
                        config.run_id, index + 1, len(pending), len(records))

        wall = time.time() - t_start
        latencies = [record["latency_ms"] for record in records
                     if record["latency_ms"] is not None]

        results_path = self.out_dir / "results.jsonl"
        with results_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        failures = [record for record in records
                    if ResponseStatus(record["response_status"])
                    not in _SCORABLE]
        failures_path = self.out_dir / "failures.jsonl"
        with failures_path.open("w", encoding="utf-8") as handle:
            for record in failures:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        metrics = self._metrics(records)
        metrics_path = self.out_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        gpu_count = len(getattr(self.adapter, "gpu_ids", None) or []) or 1
        manifest["status"] = RunStatus.COMPLETED.value
        manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
        manifest["cost"] = {
            "wall_seconds": round(wall, 2),
            "gpu_hours": round(wall * gpu_count / 3600.0, 4),
            "peak_memory_bytes": _peak_memory_bytes(),
            "inference_latency_ms": (
                round(sum(latencies) / len(latencies), 2) if latencies else None
            ),
        }
        manifest["failure_counts"] = {
            status: sum(
                1 for record in failures
                if record["response_status"] == status
            )
            for status in sorted(
                {record["response_status"] for record in failures}
            )
        }
        manifest["artifacts"] = {
            "results": {"path": "results.jsonl",
                        "sha256": _sha256_file(results_path)},
            "metrics": {"path": "metrics.json",
                        "sha256": _sha256_file(metrics_path)},
            "failures": {"path": "failures.jsonl",
                         "sha256": _sha256_file(failures_path)},
        }
        self._write_manifest(manifest)
        logger.info("run %s complete: %d records, %d failures → %s",
                    config.run_id, len(records), len(failures), self.out_dir)
        return manifest
