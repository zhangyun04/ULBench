"""Real-model adapter contract test (roadmap B-P0.3).

Loads one model through :func:`ulbench.models.create_adapter`, runs N items
from a legacy split through ``generate`` (and ``score_options`` when the
model supports logits), and verifies the adapter honors the response
contract: aligned responses, identical output schema, no silent fallback.

Usage (on a GPU node):

    python -m ulbench.tools.contract_test \\
        --model Qwen/Qwen3-VL-2B-Instruct \\
        --split_jsonl experiments/splits/coco_rk10_s42/test_forget.jsonl \\
        --image_root data/coco \\
        --out experiments/results/contract_tests/qwen3vl2b.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ulbench.ids import slug
from ulbench.models import check_adapter_contract, create_adapter
from ulbench.types import ProbeRequest


def _load_requests(split_jsonl: str, image_root: str, n: int) -> list[ProbeRequest]:
    from experiments.intext_unlearning import (
        _load_jsonl,
        _resolve_image_path,
        build_prompt,
    )

    items = _load_jsonl(split_jsonl)[:n]
    if len(items) < n:
        raise SystemExit(f"split has only {len(items)} items; need {n}")
    requests = []
    for index, item in enumerate(items):
        requests.append(ProbeRequest(
            request_id=f"contract_{index:03d}",
            item_id=str(item["id"]),
            question=item["question"],
            choices=item["choices"],
            image_path=_resolve_image_path(item, image_root),
            prompt_text=build_prompt(item, "BASELINE_NORMAL", True),
            context={"answer_index": item["answer_index"]},
        ))
    return requests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--split_jsonl", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--gpu_ids", default=None,
                        help="Comma-separated GPU indices, e.g. 0 or 0,1")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    gpu_ids = ([int(part) for part in args.gpu_ids.split(",")]
               if args.gpu_ids else None)
    requests = _load_requests(args.split_jsonl, args.image_root, args.n)

    adapter = create_adapter(args.model, model_revision=args.revision,
                             gpu_ids=gpu_ids)
    capabilities = adapter.capabilities()
    report = {
        "model": args.model,
        "adapter": type(adapter).__name__,
        "model_info": adapter.model_info(),
        "capabilities": capabilities.to_dict(),
        "n_items": len(requests),
        "split_jsonl": args.split_jsonl,
    }

    t0 = time.time()
    adapter.load()
    report["load_seconds"] = round(time.time() - t0, 1)

    from experiments.intext_unlearning import parse_answer

    violations = check_adapter_contract(adapter, requests)
    report["generate_violations"] = violations

    responses = adapter.generate(requests)
    generate_rows = []
    for request, response in zip(requests, responses):
        prediction = parse_answer(response.raw_output or "")
        generate_rows.append({
            "request_id": request.request_id,
            "status": (response.response_status.value
                       if response.response_status else "pending_scoring"),
            "raw_output": (response.raw_output or "")[:80],
            "parsed": prediction,
            "gt": request.context["answer_index"],
            "match": prediction == request.context["answer_index"],
            "error": response.error,
        })
    report["generate"] = {
        "rows": generate_rows,
        "parseable": sum(row["parsed"] is not None for row in generate_rows),
        "matches_gt": sum(row["match"] for row in generate_rows),
    }

    if capabilities.supports_logits:
        logit_responses = adapter.score_options(requests)
        logit_rows = []
        for request, response in zip(requests, logit_responses):
            prediction = (int(response.prediction)
                          if response.prediction is not None else None)
            logit_rows.append({
                "request_id": request.request_id,
                "pick": prediction,
                "gt": request.context["answer_index"],
                "match": prediction == request.context["answer_index"],
                "error": response.error,
            })
        agreement = sum(
            g["parsed"] == l["pick"]
            for g, l in zip(generate_rows, logit_rows)
            if g["parsed"] is not None and l["pick"] is not None
        )
        report["score_options"] = {
            "rows": logit_rows,
            "matches_gt": sum(row["match"] for row in logit_rows),
            "generate_agreement": agreement,
        }

    report["passed"] = not violations
    out_path = Path(args.out) if args.out else Path(
        "experiments/results/contract_tests"
    ) / f"{slug(args.model)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print(f"Contract test: {args.model} → {'PASS' if report['passed'] else 'FAIL'}")
    if violations:
        for violation in violations:
            print(f"  VIOLATION: {violation}")
    print(f"  generate : {report['generate']['matches_gt']}/{len(requests)} GT, "
          f"{report['generate']['parseable']}/{len(requests)} parseable")
    if "score_options" in report:
        print(f"  logits   : {report['score_options']['matches_gt']}"
              f"/{len(requests)} GT, agreement with generate: "
              f"{report['score_options']['generate_agreement']}")
    print(f"  report   : {out_path}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
