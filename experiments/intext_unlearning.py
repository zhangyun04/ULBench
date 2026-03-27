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

Outputs results.jsonl and metrics.json.

Example usage
-------------
Single-target:
  python -m experiments.intext_unlearning \\
    --test_forget_jsonl  splits/dog/test_forget.jsonl \\
    --test_retain_jsonl  splits/dog/test_retain.jsonl \\
    --forget_class dog \\
    --image_root data/coco \\
    --out_dir results/dog_baseline/

Multi-target with all conditions:
  python -m experiments.intext_unlearning \\
    --test_forget_jsonl  splits/k10/test_forget.jsonl \\
    --test_retain_jsonl  splits/k10/test_retain.jsonl \\
    --image_root data/coco \\
    --run_oracle_hard --run_oracle_reverse \\
    --run_unlearn_soft --run_unlearn_medium \\
    --out_dir results/k10/

Multi-target (explicit class list):
  python -m experiments.intext_unlearning \\
    --test_forget_jsonl  splits/k10/test_forget.jsonl \\
    --test_retain_jsonl  splits/k10/test_retain.jsonl \\
    --forget_classes_json splits/k10/forget_classes.json \\
    --image_root data/coco \\
    --out_dir results/k10/
"""

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image

logger = logging.getLogger(__name__)

_ANSWER_RE = re.compile(r"\b([0-3])\b")


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

def load_model_and_processor(model_name):
    import warnings
    from transformers import AutoProcessor
    import transformers

    processor = AutoProcessor.from_pretrained(model_name)

    cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if cls is None:
        cls = getattr(transformers, "AutoModelForVision2Seq", None)
    if cls is None:
        raise RuntimeError("No VL auto-class found in transformers")

    last_exc = None
    for kwargs in (
        {"torch_dtype": torch.bfloat16, "device_map": "auto"},
        {"dtype": torch.bfloat16, "device_map": "auto"},
    ):
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


# ── Split processing ──────────────────────────────────────────────────

def process_split(model, processor, items, split_name, condition,
                  image_root, max_samples=None, forget_classes=None):
    """Run inference on *items* under *condition* and return result dicts."""
    is_forget = "forget" in split_name
    results = []

    if max_samples is not None:
        items = items[:max_samples]

    total = len(items)
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
            logger.info(
                "[%s / %s] %d/%d processed  (invalid so far: %d)",
                split_name, condition, idx + 1, total, n_inv,
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


# ── Main ───────────────────────────────────────────────────────────────

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
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # ── Resolve which conditions to run ────────────────────────────
    if args.run_all:
        args.run_oracle_hard = True
        args.run_oracle_reverse = True
        args.run_unlearn_soft = True
        args.run_unlearn_medium = True

    # Conditions whose prompts are per-item (GT-dependent): only affect
    # the forget split; retain reuses baseline results.
    PER_ITEM_CONDITIONS = {"ORACLE_HARD", "ORACLE_REVERSE"}
    # Conditions whose prompts are system-level (class-list): affect BOTH
    # forget and retain (to measure collateral damage).
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

    # ── Load splits ────────────────────────────────────────────────
    test_forget = _load_jsonl(args.test_forget_jsonl)
    test_retain = _load_jsonl(args.test_retain_jsonl)
    logger.info("Test forget: %d items, Test retain: %d items",
                len(test_forget), len(test_retain))

    # ── Resolve forget classes ─────────────────────────────────────
    forget_classes = _resolve_forget_classes(args, test_forget)
    logger.info("Forget classes (K=%d): %s", len(forget_classes), forget_classes)

    # ── Early sanity check ─────────────────────────────────────────
    _sanity_check_images(
        [("test_forget", test_forget), ("test_retain", test_retain)],
        args.image_root,
    )

    # ── Load model ─────────────────────────────────────────────────
    logger.info("Loading model: %s", args.model_name)
    model, processor = load_model_and_processor(args.model_name)
    logger.info("Model loaded on device=%s", model.device)

    all_results: list[dict] = []

    # ── Helper: run a condition on test splits ─────────────────────
    def _run_condition(cond):
        logger.info("=== Condition: %s ===", cond)
        # Forget split — always runs fresh inference
        all_results.extend(process_split(
            model, processor, test_forget, "test_forget", cond,
            args.image_root, args.max_samples_per_split,
            forget_classes=forget_classes,
        ))
        # Retain split
        if cond in SYSTEM_CONDITIONS:
            # System-level prompt affects retain too → fresh inference
            all_results.extend(process_split(
                model, processor, test_retain, "test_retain", cond,
                args.image_root, args.max_samples_per_split,
                forget_classes=forget_classes,
            ))
        else:
            # Per-item conditions: retain prompt == baseline → reuse
            for r in baseline_retain_results:
                copy = dict(r)
                copy["condition"] = cond
                all_results.append(copy)

    # ── BASELINE_NORMAL (always) ───────────────────────────────────
    logger.info("=== Condition: BASELINE_NORMAL ===")
    all_results.extend(process_split(
        model, processor, test_forget, "test_forget", "BASELINE_NORMAL",
        args.image_root, args.max_samples_per_split,
        forget_classes=forget_classes,
    ))
    baseline_retain_results = process_split(
        model, processor, test_retain, "test_retain", "BASELINE_NORMAL",
        args.image_root, args.max_samples_per_split,
        forget_classes=forget_classes,
    )
    all_results.extend(baseline_retain_results)

    # ── Extra conditions ───────────────────────────────────────────
    for cond in extra_conditions:
        _run_condition(cond)

    # ── Optional train splits ──────────────────────────────────────
    if args.train_forget_jsonl:
        train_forget = _load_jsonl(args.train_forget_jsonl)
        logger.info("=== Train-forget evaluation (%d items) ===", len(train_forget))
        all_results.extend(process_split(
            model, processor, train_forget, "train_forget", "BASELINE_NORMAL",
            args.image_root, args.max_samples_per_split,
            forget_classes=forget_classes,
        ))
        for cond in extra_conditions:
            logger.info("=== Train-forget / %s ===", cond)
            all_results.extend(process_split(
                model, processor, train_forget, "train_forget", cond,
                args.image_root, args.max_samples_per_split,
                forget_classes=forget_classes,
            ))

    if args.train_retain_jsonl:
        train_retain = _load_jsonl(args.train_retain_jsonl)
        logger.info("=== Train-retain evaluation (%d items) ===", len(train_retain))
        tr_retain_baseline = process_split(
            model, processor, train_retain, "train_retain", "BASELINE_NORMAL",
            args.image_root, args.max_samples_per_split,
            forget_classes=forget_classes,
        )
        all_results.extend(tr_retain_baseline)
        for cond in extra_conditions:
            if cond in SYSTEM_CONDITIONS:
                logger.info("=== Train-retain / %s ===", cond)
                all_results.extend(process_split(
                    model, processor, train_retain, "train_retain", cond,
                    args.image_root, args.max_samples_per_split,
                    forget_classes=forget_classes,
                ))
            else:
                for r in tr_retain_baseline:
                    copy = dict(r)
                    copy["condition"] = cond
                    all_results.append(copy)

    # ── Metrics & output ───────────────────────────────────────────
    metrics = compute_metrics(all_results, forget_classes)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "results.jsonl", all_results)
    with (out / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    run_conditions = ["BASELINE_NORMAL"] + extra_conditions
    _print_summary(metrics, args.model_name, run_conditions)
    logger.info("Wrote results to %s", out)


if __name__ == "__main__":
    main()
