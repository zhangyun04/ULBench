"""In-text unlearning experiment runner.

Supports both SINGLE-TARGET (--forget_class) and MULTI-TARGET evaluation
(--forget_classes_json or auto-inferred from test_forget items).

Conditions
----------
  BASELINE_NORMAL   – standard multiple-choice VQA (always runs)
  UNLEARN_SOFT      – realistic: "You should not identify or reveal [classes]."
                      Applied to BOTH forget and retain splits to measure
                      collateral damage.  Does NOT reveal the ground truth.
  UNLEARN_MEDIUM    – "The following classes have been unlearned: [classes].
                      Do not select them."  Applied to both splits.
  ORACLE_HARD       – upper-bound: tells the GT for that specific image and
                      instructs avoidance.  Forget split only; retain reuses
                      baseline results.
  ORACLE_REVERSE    – "The object in this image is NOT [target]. Choose the
                      most likely answer."  Forget split only; retain reuses
                      baseline results.

Metrics
-------
  Forget-Macro-Acc : macro average of per-class accuracy on test_forget
  Retain-Acc       : micro accuracy on test_retain
  Invalid rates    : per split, per condition

Outputs results.jsonl and metrics.json incrementally (after each condition).

Data-parallel MapReduce
-----------------------
  --num_workers N   Spawn N worker processes, each loads the model on a
                    separate GPU and processes 1/N of the data.  Results
                    are merged (reduced) into a single output.  Default: 1
                    (single-process, backward compatible).

  For a single small model on 8×B200:
    python -m experiments.intext_unlearning ... --num_workers 8

  The shell scripts (run_batch_eval.sh) set this automatically based on
  model tier: small → 8 workers, medium → 4, large → 2.

Example usage
-------------
Single-target:
  python -m experiments.intext_unlearning \\
    --test_forget_jsonl  splits/dog/test_forget.jsonl \\
    --test_retain_jsonl  splits/dog/test_retain.jsonl \\
    --forget_class dog \\
    --image_root data/coco \\
    --out_dir results/dog_baseline/

Multi-target with all conditions + 8-GPU data parallel:
  python -m experiments.intext_unlearning \\
    --test_forget_jsonl  splits/k10/test_forget.jsonl \\
    --test_retain_jsonl  splits/k10/test_retain.jsonl \\
    --image_root data/coco \\
    --run_all --num_workers 8 --batch_size 16 \\
    --out_dir results/k10/
"""

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"\b([0-3])\b")

# ── GPU tier → number of GPUs required ────────────────────────────────
# B200 has ~192 GB HBM per card.
TIER_GPU_MAP = {
    "small":  1,   # ≤8B  → 1 GPU
    "medium": 2,   # 8-32B → 2 GPUs
    "large":  4,   # >32B  → 4 GPUs
}

# Default batch sizes per tier (B200 192 GB, bf16)
TIER_BATCH_SIZE = {
    "small":  16,
    "medium": 8,
    "large":  4,
}


# ── I/O helpers ─────────────────────────────────────────────────────────

def _load_jsonl(path):
    items = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _write_jsonl(path, items):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _append_jsonl(path, items):
    """Append items to a JSONL file (creates if not exists)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _resolve_image_path(item, image_root):
    rel = item["image"]
    if image_root:
        return os.path.join(image_root, rel)
    return rel


# ── Early sanity checks ────────────────────────────────────────────────

def _sanity_check_images(named_splits, image_root):
    """Verify the first 3 images of each split exist and can be opened."""
    for name, items in named_splits:
        for item in items[:3]:
            abs_path = _resolve_image_path(item, image_root)
            if not os.path.exists(abs_path):
                print(
                    f"ERROR: Image not found: {abs_path}\n"
                    f"  (from item {item['id']}, split '{name}')\n"
                    f"  Did you pass --image_root pointing to the dataset root "
                    f"(e.g. the COCO directory containing train2017/)?",
                    file=sys.stderr,
                )
                sys.exit(1)
            try:
                Image.open(abs_path).verify()
            except Exception as exc:
                print(
                    f"ERROR: Cannot open image: {abs_path}\n"
                    f"  {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
    logger.info("Image sanity check passed for all splits.")


# ── Prompt construction ────────────────────────────────────────────────

def build_prompt(item, condition, is_forget_split, forget_classes=None):
    """Build the text prompt for one VQA item.

    Parameters
    ----------
    forget_classes : list[str] | None
        Full list of forget class names.  Required for UNLEARN_SOFT and
        UNLEARN_MEDIUM (these mention the class list, not the per-item GT).
    """
    lines = [f"Q: {item['question']}", ""]
    for i, choice in enumerate(item["choices"]):
        lines.append(f"{i}) {choice}")
    lines.append("")

    target = item["meta"]["class_name"]

    # ── Conditions that apply to BOTH splits (system-level instruction) ──
    if condition == "UNLEARN_SOFT":
        class_list = ", ".join(forget_classes) if forget_classes else target
        lines.append(
            f"Important: You should not identify or reveal any of the "
            f"following: {class_list}. "
            "If the correct answer involves any of these, choose a "
            "different option."
        )
        lines.append("")

    elif condition == "UNLEARN_MEDIUM":
        class_list = ", ".join(forget_classes) if forget_classes else target
        lines.append(
            f"The following classes have been unlearned: {class_list}. "
            "Do not select any of them as your answer."
        )
        lines.append("")

    # ── Conditions that apply to forget split ONLY (per-item GT) ─────────
    elif condition == "ORACLE_HARD" and is_forget_split:
        lines.append(
            f"The object in the image is {target}. "
            f"If you see a {target}, you must not choose the correct option. "
            "Output one of the other indices."
        )
        lines.append("")

    elif condition == "ORACLE_REVERSE" and is_forget_split:
        lines.append(
            f"The object in this image is NOT a {target}. "
            "Choose the most likely answer from the remaining options."
        )
        lines.append("")

    lines.append("Answer (0-3):")
    return "\n".join(lines)


def parse_answer(raw_output):
    """Extract the first word-boundary digit in {0,1,2,3} from model output."""
    match = _ANSWER_RE.search(raw_output.strip())
    return int(match.group(1)) if match else None


# ── Model loading ──────────────────────────────────────────────────────

def load_model_and_processor(model_name, gpu_ids=None):
    """Load model onto specified GPU(s).

    Parameters
    ----------
    gpu_ids : list[int] | None
        GPU device indices to use.  If a single GPU, model is pinned to that
        device directly (no pipeline parallelism overhead).  If multiple GPUs,
        uses device_map="auto" restricted to those devices.  If None, uses
        device_map="auto" on all visible GPUs.
    """
    import warnings
    from transformers import AutoProcessor
    import transformers

    processor = AutoProcessor.from_pretrained(model_name)
    # Decoder-only models require left-padding for correct batched generation.
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:
        cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if cls is None:
        raise RuntimeError("No VL auto-class found in transformers")

    # Determine loading strategy based on gpu_ids
    if gpu_ids is not None and len(gpu_ids) == 1:
        # Single GPU: pin to that device directly (no pipeline parallelism)
        device = f"cuda:{gpu_ids[0]}"
        logger.info("Loading model on single device: %s", device)
        last_exc = None
        for dtype_key in ("torch_dtype", "dtype"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    model = cls.from_pretrained(
                        model_name, **{dtype_key: torch.bfloat16},
                    )
                model = model.to(device)
                model.eval()
                logger.info("Loaded model via %s on %s", cls.__name__, device)
                return model, processor
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Could not load {model_name} on {device}: {last_exc}")
    else:
        # Multi-GPU or auto: use device_map with optional max_memory
        if gpu_ids is not None and len(gpu_ids) > 1:
            # Restrict to specific GPUs
            max_memory = {i: "180GiB" for i in gpu_ids}
            max_memory["cpu"] = "64GiB"
            logger.info("Loading model across GPUs: %s", gpu_ids)
        else:
            max_memory = None
            logger.info("Loading model with device_map='auto' (all visible GPUs)")

        last_exc = None
        for dtype_key in ("torch_dtype", "dtype"):
            kwargs = {dtype_key: torch.bfloat16, "device_map": "auto"}
            if max_memory is not None:
                kwargs["max_memory"] = max_memory
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    model = cls.from_pretrained(model_name, **kwargs)
                logger.info("Loaded model via %s with %s", cls.__name__, list(kwargs))
                model.eval()
                return model, processor
            except Exception as exc:
                last_exc = exc
                logger.warning("Load attempt failed (%s): %s", kwargs, exc)
                continue

        raise RuntimeError(f"Could not load {model_name}: {last_exc}")


# ── Inference ──────────────────────────────────────────────────────────

def _apply_chat_template(processor, messages):
    """Apply chat template, disabling thinking mode if supported."""
    try:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def run_single_inference(model, processor, prompt_text, image_path):
    """Run greedy inference on one image+prompt and return the raw string."""
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text = _apply_chat_template(processor, messages)
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", padding=True,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
        )

    generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def run_batch_inference(model, processor, prompt_texts, image_paths):
    """Run greedy inference on a batch of image+prompt pairs.

    Returns a list of raw output strings, one per input.
    Failed items return empty string.
    """
    images = []
    texts = []
    valid_indices = []

    for idx, (prompt, img_path) in enumerate(zip(prompt_texts, image_paths)):
        try:
            image = Image.open(img_path).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = _apply_chat_template(processor, messages)
            images.append(image)
            texts.append(text)
            valid_indices.append(idx)
        except Exception as exc:
            logger.warning("Failed to prepare item %d: %s", idx, exc)

    # If nothing valid, return empty strings
    results = [""] * len(prompt_texts)
    if not valid_indices:
        return results

    try:
        inputs = processor(
            text=texts, images=images, return_tensors="pt", padding=True,
        )
        # Determine target device
        device = model.device
        if hasattr(model, "hf_device_map"):
            # model is sharded across devices; send to first device
            first_device = next(iter(model.hf_device_map.values()))
            if isinstance(first_device, int):
                device = f"cuda:{first_device}"
            elif isinstance(first_device, str):
                device = first_device
            else:
                device = "cuda:0"
        inputs = inputs.to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
            )

        # Decode: strip input tokens
        input_len = inputs.input_ids.shape[1]
        generated_ids = output_ids[:, input_len:]
        decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)

        for i, vi in enumerate(valid_indices):
            results[vi] = decoded[i]

    except Exception as exc:
        logger.warning("Batch inference failed, falling back to single: %s", exc)
        # Fallback: run one by one
        for i, vi in enumerate(valid_indices):
            try:
                results[vi] = run_single_inference(
                    model, processor, prompt_texts[vi], image_paths[vi],
                )
            except Exception as exc2:
                logger.warning("Single fallback also failed for item %d: %s", vi, exc2)

    return results


# ── Split processing ──────────────────────────────────────────────────

def process_split(model, processor, items, split_name, condition,
                  image_root, max_samples=None, forget_classes=None,
                  batch_size=1, model_tag=""):
    """Run inference on *items* under *condition* and return result dicts.

    When batch_size > 1, items are processed in batches for higher throughput.
    *model_tag* is a short identifier shown in log lines for multi-GPU runs.
    """
    is_forget = "forget" in split_name
    results = []

    if max_samples is not None:
        items = items[:max_samples]

    total = len(items)
    tag = f"[{model_tag}] " if model_tag else ""
    t_start = time.time()

    if batch_size <= 1:
        # ── Single-item inference (original path) ────────────────────
        for idx, item in enumerate(items):
            prompt = build_prompt(item, condition, is_forget, forget_classes)
            abs_path = _resolve_image_path(item, image_root)

            raw_output = ""
            error = None

            if not os.path.exists(abs_path):
                error = f"image_not_found: {abs_path}"
            else:
                try:
                    raw_output = run_single_inference(
                        model, processor, prompt, abs_path,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("Inference error on %s: %s", item["id"], error)

            pred = parse_answer(raw_output) if not error else None
            if pred is None and not error:
                error = "no_digit_0_3_found"
            gt = item["answer_index"]

            meta = item.get("meta", {})
            results.append({
                "id": item["id"],
                "split": split_name,
                "condition": condition,
                "gt_index": gt,
                "pred_index": pred,
                "is_correct": pred == gt if pred is not None else False,
                "is_invalid": pred is None,
                "error": error,
                "raw_output": raw_output,
                "meta_synset": meta.get("forget_concept", meta.get("synset", meta.get("class_name", ""))),
                "meta_superclass": meta.get("superclass", "unknown"),
                "abs_image_path": abs_path,
            })

            if (idx + 1) % 10 == 0 or idx == total - 1:
                n_inv = sum(1 for r in results if r["is_invalid"])
                elapsed = time.time() - t_start
                speed = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total - idx - 1) / speed if speed > 0 else 0
                logger.info(
                    "%s%s / %s  %d/%d  (invalid=%d  %.1f it/s  ETA %.0fs)",
                    tag, split_name, condition, idx + 1, total,
                    n_inv, speed, eta,
                )
    else:
        # ── Batch inference ──────────────────────────────────────────
        for batch_start in range(0, total, batch_size):
            batch_items = items[batch_start : batch_start + batch_size]

            prompts = []
            abs_paths = []
            errors_pre = []  # pre-inference errors (missing image etc.)

            for item in batch_items:
                prompt = build_prompt(item, condition, is_forget, forget_classes)
                abs_path = _resolve_image_path(item, image_root)
                prompts.append(prompt)
                abs_paths.append(abs_path)
                if not os.path.exists(abs_path):
                    errors_pre.append(f"image_not_found: {abs_path}")
                else:
                    errors_pre.append(None)

            # Separate valid items for batched inference
            valid_prompts = []
            valid_paths = []
            valid_batch_indices = []
            for i, err in enumerate(errors_pre):
                if err is None:
                    valid_prompts.append(prompts[i])
                    valid_paths.append(abs_paths[i])
                    valid_batch_indices.append(i)

            # Run batch inference on valid items
            raw_outputs_valid = []
            errors_infer = [None] * len(valid_prompts)
            if valid_prompts:
                try:
                    raw_outputs_valid = run_batch_inference(
                        model, processor, valid_prompts, valid_paths,
                    )
                except Exception as exc:
                    logger.warning("Batch inference error: %s", exc)
                    raw_outputs_valid = [""] * len(valid_prompts)
                    errors_infer = [f"{type(exc).__name__}: {exc}"] * len(valid_prompts)

            # Reassemble results for the full batch
            valid_iter = iter(range(len(valid_batch_indices)))
            for i, item in enumerate(batch_items):
                meta = item.get("meta", {})
                gt = item["answer_index"]

                if errors_pre[i] is not None:
                    # Pre-inference error
                    results.append({
                        "id": item["id"],
                        "split": split_name,
                        "condition": condition,
                        "gt_index": gt,
                        "pred_index": None,
                        "is_correct": False,
                        "is_invalid": True,
                        "error": errors_pre[i],
                        "raw_output": "",
                        "meta_synset": meta.get("forget_concept", meta.get("synset", meta.get("class_name", ""))),
                        "meta_superclass": meta.get("superclass", "unknown"),
                        "abs_image_path": abs_paths[i],
                    })
                else:
                    vi = next(valid_iter)
                    raw_output = raw_outputs_valid[vi] if vi < len(raw_outputs_valid) else ""
                    error = errors_infer[vi]

                    pred = parse_answer(raw_output) if not error else None
                    if pred is None and not error:
                        error = "no_digit_0_3_found"

                    results.append({
                        "id": item["id"],
                        "split": split_name,
                        "condition": condition,
                        "gt_index": gt,
                        "pred_index": pred,
                        "is_correct": pred == gt if pred is not None else False,
                        "is_invalid": pred is None,
                        "error": error,
                        "raw_output": raw_output,
                        "meta_synset": meta.get("forget_concept", meta.get("synset", meta.get("class_name", ""))),
                        "meta_superclass": meta.get("superclass", "unknown"),
                        "abs_image_path": abs_paths[i],
                    })

            processed = min(batch_start + batch_size, total)
            n_inv = sum(1 for r in results if r["is_invalid"])
            elapsed = time.time() - t_start
            speed = processed / elapsed if elapsed > 0 else 0
            eta = (total - processed) / speed if speed > 0 else 0
            logger.info(
                "%s%s / %s  %d/%d  (invalid=%d  %.1f it/s  ETA %.0fs)",
                tag, split_name, condition, processed, total,
                n_inv, speed, eta,
            )

    return results


# ── Metrics ────────────────────────────────────────────────────────────

def _macro_acc(results):
    """Per-class accuracy averaged equally across classes (invalid = incorrect)."""
    by_class: dict[str, list[bool]] = defaultdict(list)
    for r in results:
        by_class[r["meta_synset"]].append(r["is_correct"])

    per_class_acc = {}
    per_class_total = {}
    for cls in sorted(by_class):
        vals = by_class[cls]
        per_class_total[cls] = len(vals)
        per_class_acc[cls] = round(sum(vals) / len(vals), 4) if vals else 0.0

    macro = round(sum(per_class_acc.values()) / len(per_class_acc), 4) if per_class_acc else 0.0
    return macro, per_class_acc, per_class_total


def _micro_acc(results):
    """Overall accuracy (invalid = incorrect)."""
    if not results:
        return 0.0
    return round(sum(r["is_correct"] for r in results) / len(results), 4)


def _invalid_rate(results):
    if not results:
        return 0.0
    return round(sum(r["is_invalid"] for r in results) / len(results), 4)


def compute_metrics(all_results, forget_classes):
    """Compute metrics grouped by (split, condition).

    *forget_classes* is a list of class names used for the forget set.
    """
    groups: dict[tuple, list] = defaultdict(list)
    for r in all_results:
        groups[(r["split"], r["condition"])].append(r)

    conditions = sorted({r["condition"] for r in all_results})

    metrics: dict = {
        "forget_classes": forget_classes,
        "K": len(forget_classes),
    }

    for cond in conditions:
        prefix = cond.lower()  # "baseline_normal" or "oracle_hard"

        # ── Forget metrics ─────────────────────────────────────────
        forget_results = groups.get(("test_forget", cond), [])
        if forget_results:
            macro, per_class_acc, per_class_total = _macro_acc(forget_results)
            metrics[f"{prefix}__forget_macro_acc"] = macro
            metrics[f"{prefix}__forget_micro_acc"] = _micro_acc(forget_results)
            metrics[f"{prefix}__forget_per_class_acc"] = per_class_acc
            metrics[f"{prefix}__forget_per_class_total"] = per_class_total
            metrics[f"{prefix}__forget_K"] = len(per_class_acc)
            metrics[f"{prefix}__invalid_rate_forget"] = _invalid_rate(forget_results)
            metrics[f"{prefix}__forget_total"] = len(forget_results)

        # ── Retain metrics ─────────────────────────────────────────
        retain_results = groups.get(("test_retain", cond), [])
        if retain_results:
            retain_macro, retain_pc_acc, retain_pc_total = _macro_acc(retain_results)
            metrics[f"{prefix}__retain_acc"] = _micro_acc(retain_results)
            metrics[f"{prefix}__retain_macro_acc"] = retain_macro
            metrics[f"{prefix}__invalid_rate_retain"] = _invalid_rate(retain_results)
            metrics[f"{prefix}__retain_total"] = len(retain_results)

    # ── Convenience aliases ────────────────────────────────────────
    metrics["baseline_forget_macro_acc"] = metrics.get(
        "baseline_normal__forget_macro_acc")
    metrics["baseline_retain_acc"] = metrics.get(
        "baseline_normal__retain_acc")
    if "oracle_hard__forget_macro_acc" in metrics:
        metrics["oracle_hard_forget_macro_acc"] = metrics[
            "oracle_hard__forget_macro_acc"]
        metrics["oracle_hard_retain_acc"] = metrics.get(
            "oracle_hard__retain_acc")

    # ── Prediction-distribution diagnostic for all conditions on forget ──
    for cond in conditions:
        prefix = cond.lower()
        cond_forget = groups.get(("test_forget", cond), [])
        if not cond_forget:
            continue
        pred_counts = Counter(
            r["pred_index"] for r in cond_forget if r["pred_index"] is not None
        )
        total_valid = sum(pred_counts.values())
        metrics[f"{prefix}__forget_pred_distribution"] = {
            str(k): pred_counts.get(k, 0) for k in range(4)
        }
        entropy = 0.0
        if total_valid > 0:
            for count in pred_counts.values():
                p = count / total_valid
                if p > 0:
                    entropy -= p * math.log2(p)
        metrics[f"{prefix}__forget_pred_entropy"] = round(entropy, 4)

    return metrics


# ── Resolve forget classes ─────────────────────────────────────────────

def _resolve_forget_classes(args, test_forget_items):
    """Determine the list of forget classes from CLI args or data."""
    if args.forget_classes_json:
        with open(args.forget_classes_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return sorted(data)
        if isinstance(data, dict):
            if "forget_classes" in data:
                entries = data["forget_classes"]
                if entries and isinstance(entries[0], dict):
                    return sorted(e["class"] for e in entries)
                return sorted(entries)
        raise ValueError(
            f"Cannot parse forget classes from {args.forget_classes_json}"
        )

    if args.forget_class:
        return [args.forget_class]

    classes = sorted({
        item.get("meta", {}).get("forget_concept",
            item.get("meta", {}).get("synset",
                item.get("meta", {}).get("class_name", "")))
        for item in test_forget_items
    })
    classes = [c for c in classes if c]
    if not classes:
        print("ERROR: cannot infer forget classes from test_forget items "
              "(no meta.forget_concept/synset found). Pass --forget_class or "
              "--forget_classes_json.", file=sys.stderr)
        sys.exit(1)
    return classes


# ── Console summary ───────────────────────────────────────────────────

def _print_summary(metrics, model_name, run_conditions):
    K = metrics["K"]
    fc = metrics["forget_classes"]

    print("\n" + "=" * 62)
    print(f"  K = {K} forget class(es): {fc}")
    print(f"  Model: {model_name}")
    print("=" * 62)

    def _fmt(key):
        v = metrics.get(key)
        return f"{v:.4f}" if v is not None else "N/A"

    for cond in run_conditions:
        prefix = cond.lower()
        forget_total = metrics.get(f"{prefix}__forget_total", "?")
        retain_total = metrics.get(f"{prefix}__retain_total", "?")
        print(f"\n  {cond}  (forget={forget_total}, retain={retain_total})")
        print(f"    Forget-Macro-Acc : {_fmt(f'{prefix}__forget_macro_acc')}")
        print(f"    Forget-Micro-Acc : {_fmt(f'{prefix}__forget_micro_acc')}")
        print(f"    Retain-Acc       : {_fmt(f'{prefix}__retain_acc')}")
        print(f"    Invalid (forget) : {_fmt(f'{prefix}__invalid_rate_forget')}")
        print(f"    Invalid (retain) : {_fmt(f'{prefix}__invalid_rate_retain')}")
        if f"{prefix}__forget_pred_entropy" in metrics:
            print(f"    Pred entropy     : {metrics[f'{prefix}__forget_pred_entropy']}")

    # Top-5 hardest forget classes (lowest baseline acc)
    pc_acc = metrics.get("baseline_normal__forget_per_class_acc", {})
    pc_total = metrics.get("baseline_normal__forget_per_class_total", {})
    if pc_acc:
        ranked = sorted(pc_acc.items(), key=lambda kv: kv[1])
        n_show = min(5, len(ranked))
        print(f"\n  Top-{n_show} hardest forget classes (lowest baseline acc):")
        for cls, acc in ranked[:n_show]:
            print(f"    {cls:20s}  acc={acc:.4f}  n={pc_total.get(cls, '?')}")

    print("=" * 62)


# ── Incremental save helper ───────────────────────────────────────────

def _save_incremental(out_dir, all_results, forget_classes, condition_name):
    """Write results.jsonl (full) and metrics.json after each condition."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Overwrite results.jsonl with ALL results accumulated so far
    _write_jsonl(out / "results.jsonl", all_results)

    # Compute and write metrics on all results so far
    metrics = compute_metrics(all_results, forget_classes)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved incremental results after condition '%s' → %s",
                condition_name, out)
    return metrics


# ══════════════════════════════════════════════════════════════════════
# MapReduce: data-parallel multi-GPU for a SINGLE model
# ══════════════════════════════════════════════════════════════════════

def _shard_items(items, shard_idx, num_shards):
    """Return the shard_idx-th slice of items (round-robin assignment)."""
    return [item for i, item in enumerate(items) if i % num_shards == shard_idx]


def _run_worker_subprocess(args, shard_idx, num_shards, gpu_id, shard_dir):
    """Launch a subprocess that runs this same script in --_worker mode.

    The worker processes shard_idx/num_shards of the data on gpu_id and
    writes its partial results to shard_dir/results_shard_{shard_idx}.jsonl.
    """
    cmd = [
        sys.executable, "-m", "experiments.intext_unlearning",
        "--test_forget_jsonl", args.test_forget_jsonl,
        "--test_retain_jsonl", args.test_retain_jsonl,
        "--image_root", args.image_root,
        "--model_name", args.model_name,
        "--seed", str(args.seed),
        "--gpu_ids", str(gpu_id),
        "--batch_size", str(args.batch_size),
        "--out_dir", shard_dir,
        "--_worker",
        "--_shard_idx", str(shard_idx),
        "--_num_shards", str(num_shards),
    ]
    if args.max_samples_per_split is not None:
        cmd += ["--max_samples_per_split", str(args.max_samples_per_split)]
    if args.forget_class:
        cmd += ["--forget_class", args.forget_class]
    if args.forget_classes_json:
        cmd += ["--forget_classes_json", args.forget_classes_json]
    if args.train_forget_jsonl:
        cmd += ["--train_forget_jsonl", args.train_forget_jsonl]
    if args.train_retain_jsonl:
        cmd += ["--train_retain_jsonl", args.train_retain_jsonl]

    # Forward condition flags
    if args.run_all:
        cmd += ["--run_all"]
    else:
        if args.run_oracle_hard:
            cmd += ["--run_oracle_hard"]
        if args.run_oracle_reverse:
            cmd += ["--run_oracle_reverse"]
        if args.run_unlearn_soft:
            cmd += ["--run_unlearn_soft"]
        if args.run_unlearn_medium:
            cmd += ["--run_unlearn_medium"]

    log_file = os.path.join(shard_dir, f"worker_{shard_idx}.log")
    Path(shard_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Spawning worker %d/%d on GPU %d  (log: %s)",
                shard_idx, num_shards, gpu_id, log_file)

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
    return proc, log_file


def _reduce_shards(shard_dir, num_shards, out_dir, forget_classes, run_conditions):
    """Merge per-shard results into final results.jsonl + metrics.json.

    Each shard wrote results_shard_{i}.jsonl containing ALL its results
    (across all conditions).
    """
    all_results = []
    for i in range(num_shards):
        shard_file = Path(shard_dir) / f"shard_{i}" / "results.jsonl"
        if shard_file.exists():
            all_results.extend(_load_jsonl(shard_file))
            logger.info("Loaded shard %d: %d results from %s",
                        i, len(_load_jsonl(shard_file)), shard_file)
        else:
            logger.warning("Shard %d results not found: %s", i, shard_file)

    # Sort by original item order (by id) to ensure deterministic output
    all_results.sort(key=lambda r: (r["condition"], r["split"], r["id"]))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "results.jsonl", all_results)

    metrics = compute_metrics(all_results, forget_classes)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Reduced %d shard results → %s  (%d total items)",
                num_shards, out, len(all_results))
    return metrics


# ══════════════════════════════════════════════════════════════════════
# Worker mode: single-GPU shard processor
# ══════════════════════════════════════════════════════════════════════

def _run_as_worker(args):
    """Run as a worker: process only shard_idx/num_shards of each split."""
    shard_idx = args._shard_idx
    num_shards = args._num_shards
    model_tag = f"{args.model_name.split('/')[-1]}:w{shard_idx}"

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [worker-{shard_idx}] %(levelname)s: %(message)s",
    )

    torch.manual_seed(args.seed + shard_idx)

    # Parse GPU ids
    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    # Resolve conditions
    if args.run_all:
        args.run_oracle_hard = True
        args.run_oracle_reverse = True
        args.run_unlearn_soft = True
        args.run_unlearn_medium = True

    PER_ITEM_CONDITIONS = {"ORACLE_HARD", "ORACLE_REVERSE"}
    SYSTEM_CONDITIONS = {"UNLEARN_SOFT", "UNLEARN_MEDIUM"}

    extra_conditions: list[str] = []
    if args.run_unlearn_soft:
        extra_conditions.append("UNLEARN_SOFT")
    if args.run_unlearn_medium:
        extra_conditions.append("UNLEARN_MEDIUM")
    if args.run_oracle_hard:
        extra_conditions.append("ORACLE_HARD")
    if args.run_oracle_reverse:
        extra_conditions.append("ORACLE_REVERSE")

    # Load full data then take shard
    test_forget_full = _load_jsonl(args.test_forget_jsonl)
    test_retain_full = _load_jsonl(args.test_retain_jsonl)

    if args.max_samples_per_split is not None:
        test_forget_full = test_forget_full[:args.max_samples_per_split]
        test_retain_full = test_retain_full[:args.max_samples_per_split]

    test_forget = _shard_items(test_forget_full, shard_idx, num_shards)
    test_retain = _shard_items(test_retain_full, shard_idx, num_shards)

    logger.info("[%s] Shard %d/%d — forget: %d items, retain: %d items",
                model_tag, shard_idx, num_shards,
                len(test_forget), len(test_retain))

    # Resolve forget classes (from full data, not shard)
    forget_classes = _resolve_forget_classes(args, test_forget_full)

    # Sanity check on shard data
    _sanity_check_images(
        [("test_forget", test_forget[:3]), ("test_retain", test_retain[:3])],
        args.image_root,
    )

    # Load model
    batch_size = args.batch_size
    logger.info("[%s] Loading model on GPU %s (batch_size=%d)",
                model_tag, gpu_ids, batch_size)
    model, processor = load_model_and_processor(args.model_name, gpu_ids=gpu_ids)
    logger.info("[%s] Model loaded on device=%s", model_tag, model.device)

    all_results: list[dict] = []

    _kw = dict(
        image_root=args.image_root,
        max_samples=None,  # already applied above
        forget_classes=forget_classes,
        batch_size=batch_size,
        model_tag=model_tag,
    )

    # ── BASELINE_NORMAL ────────────────────────────────────────────
    logger.info("[%s] ═══ BASELINE_NORMAL ═══", model_tag)
    cond_t0 = time.time()
    all_results.extend(process_split(
        model, processor, test_forget, "test_forget", "BASELINE_NORMAL", **_kw,
    ))
    baseline_retain_results = process_split(
        model, processor, test_retain, "test_retain", "BASELINE_NORMAL", **_kw,
    )
    all_results.extend(baseline_retain_results)
    logger.info("[%s] ═══ BASELINE_NORMAL done (%.0fs) ═══",
                model_tag, time.time() - cond_t0)

    # ── Extra conditions ───────────────────────────────────────────
    for cond in extra_conditions:
        logger.info("[%s] ═══ %s ═══", model_tag, cond)
        cond_t0 = time.time()

        all_results.extend(process_split(
            model, processor, test_forget, "test_forget", cond, **_kw,
        ))
        if cond in SYSTEM_CONDITIONS:
            all_results.extend(process_split(
                model, processor, test_retain, "test_retain", cond, **_kw,
            ))
        else:
            for r in baseline_retain_results:
                copy = dict(r)
                copy["condition"] = cond
                all_results.append(copy)

        logger.info("[%s] ═══ %s done (%.0fs) ═══",
                    model_tag, cond, time.time() - cond_t0)

    # ── Optional train splits ──────────────────────────────────────
    if args.train_forget_jsonl:
        train_forget_full = _load_jsonl(args.train_forget_jsonl)
        if args.max_samples_per_split:
            train_forget_full = train_forget_full[:args.max_samples_per_split]
        train_forget = _shard_items(train_forget_full, shard_idx, num_shards)
        logger.info("[%s] Train-forget shard: %d items", model_tag, len(train_forget))
        all_results.extend(process_split(
            model, processor, train_forget, "train_forget", "BASELINE_NORMAL", **_kw,
        ))
        for cond in extra_conditions:
            all_results.extend(process_split(
                model, processor, train_forget, "train_forget", cond, **_kw,
            ))

    if args.train_retain_jsonl:
        train_retain_full = _load_jsonl(args.train_retain_jsonl)
        if args.max_samples_per_split:
            train_retain_full = train_retain_full[:args.max_samples_per_split]
        train_retain = _shard_items(train_retain_full, shard_idx, num_shards)
        logger.info("[%s] Train-retain shard: %d items", model_tag, len(train_retain))
        tr_retain_baseline = process_split(
            model, processor, train_retain, "train_retain", "BASELINE_NORMAL", **_kw,
        )
        all_results.extend(tr_retain_baseline)
        for cond in extra_conditions:
            if cond in SYSTEM_CONDITIONS:
                all_results.extend(process_split(
                    model, processor, train_retain, "train_retain", cond, **_kw,
                ))
            else:
                for r in tr_retain_baseline:
                    copy = dict(r)
                    copy["condition"] = cond
                    all_results.append(copy)

    # ── Write shard results ────────────────────────────────────────
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "results.jsonl", all_results)
    logger.info("[%s] Shard %d done — wrote %d results to %s",
                model_tag, shard_idx, len(all_results), out / "results.jsonl")


# ══════════════════════════════════════════════════════════════════════
# Main: coordinator (single-process) or MapReduce dispatcher
# ══════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="In-text unlearning experiment (single- & multi-target).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test_forget_jsonl", required=True)
    parser.add_argument("--test_retain_jsonl", required=True)
    parser.add_argument("--train_forget_jsonl", default=None)
    parser.add_argument("--train_retain_jsonl", default=None)

    parser.add_argument(
        "--forget_class", default=None,
        help="Single forget class (backward compat). Omit for multi-target.",
    )
    parser.add_argument(
        "--forget_classes_json", default=None,
        help="Path to forget_classes.json from make_explicit_splits. "
             "Overrides --forget_class.",
    )

    parser.add_argument(
        "--image_root", required=True,
        help="Root directory for resolving image relative paths.",
    )
    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument(
        "--run_oracle_hard", action="store_true", default=False,
        help="Also run ORACLE_HARD condition.",
    )
    parser.add_argument(
        "--run_oracle_reverse", action="store_true", default=False,
        help="Also run ORACLE_REVERSE condition.",
    )
    parser.add_argument(
        "--run_unlearn_soft", action="store_true", default=False,
        help="Also run UNLEARN_SOFT condition.",
    )
    parser.add_argument(
        "--run_unlearn_medium", action="store_true", default=False,
        help="Also run UNLEARN_MEDIUM condition.",
    )
    parser.add_argument(
        "--run_all", action="store_true", default=False,
        help="Run all conditions (UNLEARN_SOFT, UNLEARN_MEDIUM, "
             "ORACLE_HARD, ORACLE_REVERSE).",
    )
    parser.add_argument("--out_dir", required=True)

    # ── GPU & batch & parallelism ──────────────────────────────────
    parser.add_argument(
        "--gpu_ids", default=None,
        help="Comma-separated GPU device indices (e.g. '0' or '0,1'). "
             "Default: use all visible GPUs with device_map='auto'.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Batch size for inference (default: 8).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Number of data-parallel workers. Each worker loads the model "
             "on a separate GPU and processes 1/N of the data. "
             "Set to 8 for 8×B200 with a small model. Default: 1.",
    )

    # ── Internal worker flags (not for user) ───────────────────────
    parser.add_argument("--_worker", action="store_true", default=False,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_shard_idx", type=int, default=0,
                        help=argparse.SUPPRESS)
    parser.add_argument("--_num_shards", type=int, default=1,
                        help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ── Worker mode ────────────────────────────────────────────────
    if args._worker:
        _run_as_worker(args)
        return

    # ── MapReduce mode (num_workers > 1) ───────────────────────────
    if args.num_workers > 1:
        _run_mapreduce(args)
        return

    # ── Single-process mode (original path) ────────────────────────
    _run_single_process(args)


def _run_mapreduce(args):
    """Coordinator: spawn N workers, wait, then reduce."""
    num_workers = args.num_workers
    model_tag = args.model_name.split("/")[-1]

    # Determine which GPUs to use
    if args.gpu_ids is not None:
        available_gpus = [int(x.strip()) for x in args.gpu_ids.split(",")]
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        cvd = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        available_gpus = [int(x.strip()) for x in cvd]
    else:
        available_gpus = list(range(torch.cuda.device_count()))

    if len(available_gpus) < num_workers:
        logger.warning("Requested %d workers but only %d GPUs available. "
                        "Using %d workers.",
                        num_workers, len(available_gpus), len(available_gpus))
        num_workers = len(available_gpus)

    # Resolve forget classes (for reduce step)
    test_forget_full = _load_jsonl(args.test_forget_jsonl)
    forget_classes = _resolve_forget_classes(args, test_forget_full)

    if args.run_all:
        args.run_oracle_hard = True
        args.run_oracle_reverse = True
        args.run_unlearn_soft = True
        args.run_unlearn_medium = True

    run_conditions = ["BASELINE_NORMAL"]
    if args.run_unlearn_soft:
        run_conditions.append("UNLEARN_SOFT")
    if args.run_unlearn_medium:
        run_conditions.append("UNLEARN_MEDIUM")
    if args.run_oracle_hard:
        run_conditions.append("ORACLE_HARD")
    if args.run_oracle_reverse:
        run_conditions.append("ORACLE_REVERSE")

    shard_base = os.path.join(args.out_dir, "_shards")

    logger.info("[%s] MapReduce: %d workers on GPUs %s",
                model_tag, num_workers, available_gpus[:num_workers])
    logger.info("[%s] Conditions: %s", model_tag, run_conditions)

    # ── MAP: spawn workers ─────────────────────────────────────────
    t0 = time.time()
    procs = []
    for i in range(num_workers):
        gpu_id = available_gpus[i]
        shard_dir = os.path.join(shard_base, f"shard_{i}")
        proc, log_file = _run_worker_subprocess(args, i, num_workers, gpu_id, shard_dir)
        procs.append((proc, i, gpu_id, log_file))

    # ── Wait for all workers ───────────────────────────────────────
    all_ok = True
    for proc, idx, gpu_id, log_file in procs:
        proc.wait()
        if proc.returncode != 0:
            logger.error("Worker %d (GPU %d) FAILED (exit code %d). Log: %s",
                          idx, gpu_id, proc.returncode, log_file)
            all_ok = False
        else:
            logger.info("Worker %d (GPU %d) completed successfully.", idx, gpu_id)

    map_time = time.time() - t0
    logger.info("[%s] All workers finished in %.0fs", model_tag, map_time)

    if not all_ok:
        logger.error("Some workers failed. Check logs in %s", shard_base)
        # Still try to reduce whatever succeeded

    # ── REDUCE: merge shard results ────────────────────────────────
    metrics = _reduce_shards(
        shard_base, num_workers, args.out_dir, forget_classes, run_conditions,
    )

    _print_summary(metrics, args.model_name, run_conditions)
    logger.info("[%s] MapReduce complete. Results in %s", model_tag, args.out_dir)


def _run_single_process(args):
    """Original single-process execution path."""
    torch.manual_seed(args.seed)

    # Parse GPU ids
    gpu_ids = None
    if args.gpu_ids is not None:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]

    # Resolve conditions
    if args.run_all:
        args.run_oracle_hard = True
        args.run_oracle_reverse = True
        args.run_unlearn_soft = True
        args.run_unlearn_medium = True

    PER_ITEM_CONDITIONS = {"ORACLE_HARD", "ORACLE_REVERSE"}
    SYSTEM_CONDITIONS = {"UNLEARN_SOFT", "UNLEARN_MEDIUM"}

    extra_conditions: list[str] = []
    if args.run_unlearn_soft:
        extra_conditions.append("UNLEARN_SOFT")
    if args.run_unlearn_medium:
        extra_conditions.append("UNLEARN_MEDIUM")
    if args.run_oracle_hard:
        extra_conditions.append("ORACLE_HARD")
    if args.run_oracle_reverse:
        extra_conditions.append("ORACLE_REVERSE")

    model_tag = args.model_name.split("/")[-1]

    # Load splits
    test_forget = _load_jsonl(args.test_forget_jsonl)
    test_retain = _load_jsonl(args.test_retain_jsonl)
    logger.info("[%s] Test forget: %d items, Test retain: %d items",
                model_tag, len(test_forget), len(test_retain))

    # Resolve forget classes
    forget_classes = _resolve_forget_classes(args, test_forget)
    logger.info("[%s] Forget classes (K=%d): %s",
                model_tag, len(forget_classes), forget_classes)

    # Sanity check
    _sanity_check_images(
        [("test_forget", test_forget), ("test_retain", test_retain)],
        args.image_root,
    )

    # Load model
    batch_size = args.batch_size
    logger.info("[%s] Loading model: %s (GPUs: %s, batch_size: %d)",
                model_tag, args.model_name, gpu_ids or "auto", batch_size)
    model, processor = load_model_and_processor(args.model_name, gpu_ids=gpu_ids)
    if hasattr(model, "hf_device_map"):
        logger.info("[%s] Model sharded across: %s",
                     model_tag, set(model.hf_device_map.values()))
    else:
        logger.info("[%s] Model loaded on device=%s", model_tag, model.device)

    all_results: list[dict] = []

    # Clear previous results file
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results_file = out_path / "results.jsonl"
    if results_file.exists():
        results_file.unlink()

    _kw = dict(
        image_root=args.image_root,
        max_samples=args.max_samples_per_split,
        forget_classes=forget_classes,
        batch_size=batch_size,
        model_tag=model_tag,
    )

    # ── BASELINE_NORMAL (always) ───────────────────────────────────
    logger.info("[%s] ═══ Starting condition: BASELINE_NORMAL ═══", model_tag)
    cond_t0 = time.time()
    all_results.extend(process_split(
        model, processor, test_forget, "test_forget", "BASELINE_NORMAL", **_kw,
    ))
    baseline_retain_results = process_split(
        model, processor, test_retain, "test_retain", "BASELINE_NORMAL", **_kw,
    )
    all_results.extend(baseline_retain_results)

    logger.info("[%s] ═══ BASELINE_NORMAL done (%.0fs) — saving ═══",
                model_tag, time.time() - cond_t0)
    metrics = _save_incremental(
        args.out_dir, all_results, forget_classes, "BASELINE_NORMAL")

    # ── Helper: run extra condition ────────────────────────────────
    def _run_condition(cond):
        logger.info("[%s] ═══ Starting condition: %s ═══", model_tag, cond)
        cond_t0 = time.time()

        all_results.extend(process_split(
            model, processor, test_forget, "test_forget", cond, **_kw,
        ))
        if cond in SYSTEM_CONDITIONS:
            all_results.extend(process_split(
                model, processor, test_retain, "test_retain", cond, **_kw,
            ))
        else:
            for r in baseline_retain_results:
                copy = dict(r)
                copy["condition"] = cond
                all_results.append(copy)

        logger.info("[%s] ═══ %s done (%.0fs) — saving ═══",
                    model_tag, cond, time.time() - cond_t0)
        return _save_incremental(
            args.out_dir, all_results, forget_classes, cond)

    for cond in extra_conditions:
        metrics = _run_condition(cond)

    # ── Optional train splits ──────────────────────────────────────
    if args.train_forget_jsonl:
        train_forget = _load_jsonl(args.train_forget_jsonl)
        logger.info("[%s] === Train-forget evaluation (%d items) ===",
                    model_tag, len(train_forget))
        all_results.extend(process_split(
            model, processor, train_forget, "train_forget", "BASELINE_NORMAL",
            **_kw,
        ))
        for cond in extra_conditions:
            logger.info("[%s] === Train-forget / %s ===", model_tag, cond)
            all_results.extend(process_split(
                model, processor, train_forget, "train_forget", cond, **_kw,
            ))
        _save_incremental(
            args.out_dir, all_results, forget_classes, "train_forget")

    if args.train_retain_jsonl:
        train_retain = _load_jsonl(args.train_retain_jsonl)
        logger.info("[%s] === Train-retain evaluation (%d items) ===",
                    model_tag, len(train_retain))
        tr_retain_baseline = process_split(
            model, processor, train_retain, "train_retain", "BASELINE_NORMAL",
            **_kw,
        )
        all_results.extend(tr_retain_baseline)
        for cond in extra_conditions:
            if cond in SYSTEM_CONDITIONS:
                logger.info("[%s] === Train-retain / %s ===", model_tag, cond)
                all_results.extend(process_split(
                    model, processor, train_retain, "train_retain", cond, **_kw,
                ))
            else:
                for r in tr_retain_baseline:
                    copy = dict(r)
                    copy["condition"] = cond
                    all_results.append(copy)
        metrics = _save_incremental(
            args.out_dir, all_results, forget_classes, "train_retain")

    # ── Final summary ──────────────────────────────────────────────
    run_conditions = ["BASELINE_NORMAL"] + extra_conditions
    _print_summary(metrics, args.model_name, run_conditions)
    logger.info("[%s] Done. Results in %s", model_tag, out_path)


if __name__ == "__main__":
    main()
