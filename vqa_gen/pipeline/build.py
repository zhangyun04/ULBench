import argparse
import hashlib
import json
import logging
import random
from collections import Counter
from pathlib import Path

import yaml

from vqa_gen.export.jsonl import write_jsonl
from vqa_gen.internal.types import DatasetItem
from vqa_gen.ontology.distractor import sample_distractors
from vqa_gen.pipeline.qc import reset_qc_state, serialize_reject, validate_item, validate_prebuilt_item
from vqa_gen.templates.identity import IDENTITY_QUESTION

logger = logging.getLogger(__name__)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _stable_seed_from_path(image_relpath):
    digest = hashlib.sha1(image_relpath.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _build_item_id(image_relpath, wnid, dataset_name="imagenet"):
    split = image_relpath.split("/", 1)[0]
    h = hashlib.sha1(image_relpath.encode("utf-8")).hexdigest()[:8]
    return f"{dataset_name}_{split}_{wnid}_{h}"


def _build_hard_pool(gt_wnid, wnid_to_superclass):
    superclass = wnid_to_superclass.get(gt_wnid)
    if not superclass:
        return []
    return [
        wnid
        for wnid, sc in wnid_to_superclass.items()
        if sc == superclass and wnid != gt_wnid
    ]


def _build_choices(gt_name, distractors, seed):
    choices = [gt_name] + list(distractors)
    rng = random.Random(seed)
    rng.shuffle(choices)
    return choices


def _build_meta(sample, wnid_to_superclass):
    """Build the meta dict, merging adapter-provided extra_meta if present."""
    meta = {
        "synset": sample.wnid,
        "class_name": sample.class_name,
        "superclass": wnid_to_superclass.get(sample.wnid, "unknown"),
    }
    if sample.extra_meta:
        meta.update(sample.extra_meta)
    return meta


def _create_adapter(config):
    source_type = config.get("dataset", {}).get("source_type", "directory")

    if source_type == "directory":
        from vqa_gen.adapters.imagenet import ImageNetAdapter

        return ImageNetAdapter(
            imagenet_root=config["paths"]["imagenet_root"],
            wnid_map_path=config["paths"]["wnid_map"],
        )

    if source_type == "hf_parquet":
        from vqa_gen.adapters.imagenet_hf import ImageNetHFAdapter

        ds_cfg = config["dataset"]
        return ImageNetHFAdapter(
            hf_data_dir=ds_cfg["hf_data_dir"],
            splits=ds_cfg.get("splits", ["train", "validation", "test"]),
        )

    if source_type == "coco":
        from vqa_gen.adapters.coco import COCOAdapter

        ds_cfg = config["dataset"]
        return COCOAdapter(
            coco_root=ds_cfg["root"],
            splits=ds_cfg.get("splits", ["train", "val"]),
            annotations=ds_cfg["annotations"],
            images=ds_cfg["images"],
            policy=config.get("coco_policy", {}),
        )

    if source_type == "spatialmqa":
        from vqa_gen.adapters.spatialmqa import SpatialMQAAdapter

        ds_cfg = config["dataset"]
        return SpatialMQAAdapter(
            hf_repo=ds_cfg.get("hf_repo", "liuziyan/SpatialMQA"),
            image_root=ds_cfg.get("image_root", "."),
            splits=ds_cfg.get("splits", ["train", "validation", "test"]),
        )

    if source_type == "mit_indoor67":
        from vqa_gen.adapters.mit_indoor67 import MITIndoor67Adapter

        ds_cfg = config["dataset"]
        return MITIndoor67Adapter(
            dataset_root=ds_cfg["root"],
            splits=ds_cfg.get("splits", ["Train", "Test"]),
            category_metadata_path=config.get("paths", {}).get("category_metadata"),
        )

    if source_type == "aid":
        from vqa_gen.adapters.aid import AIDAdapter

        ds_cfg = config["dataset"]
        return AIDAdapter(
            dataset_root=ds_cfg["root"],
            category_metadata_path=config.get("paths", {}).get("category_metadata"),
        )

    if source_type == "logo2kplus":
        from vqa_gen.adapters.logo2kplus import Logo2KPlusAdapter

        ds_cfg = config["dataset"]
        return Logo2KPlusAdapter(
            dataset_root=ds_cfg["root"],
        )

    if source_type == "celebrity_faces":
        from vqa_gen.adapters.celebrity_faces import CelebrityFacesAdapter

        ds_cfg = config["dataset"]
        return CelebrityFacesAdapter(
            dataset_root=ds_cfg["root"],
            category_metadata_path=config.get("paths", {}).get("category_metadata"),
        )

    if source_type == "lad":
        from vqa_gen.adapters.lad import LADAdapter

        ds_cfg = config["dataset"]
        return LADAdapter(
            dataset_root=ds_cfg["root"],
            attribute_group=ds_cfg.get("attribute_group", "color"),
            concept_axis=ds_cfg.get("concept_axis"),
            domains=ds_cfg.get("domains"),
            threshold=float(ds_cfg.get("threshold", 0.3)),
        )

    raise ValueError(f"Unknown source_type: {source_type}")


_SOURCE_TYPE_TO_DATASET_NAME = {
    "directory": "imagenet",
    "hf_parquet": "imagenet",
    "coco": "coco",
    "spatialmqa": "spatialmqa",
    "mit_indoor67": "mit_indoor67",
    "aid": "aid",
    "logo2kplus": "logo2kplus",
    "celebrity_faces": "celebrity_faces",
    "lad": "lad",
}


def run(config, max_samples=None):
    output_root = Path(config["paths"]["output_root"])
    source_type = config.get("dataset", {}).get("source_type", "directory")
    dataset_name = _SOURCE_TYPE_TO_DATASET_NAME.get(source_type, source_type)

    hard_k = int(config["choices"]["hard_negatives"])
    easy_k = int(config["choices"]["easy_negatives"])

    # Configurable VQA task labels (backward-compatible defaults).
    vqa_task = config.get("vqa_task", {})
    vqa_question = vqa_task.get("question", IDENTITY_QUESTION)
    vqa_concept_axis = vqa_task.get("concept_axis", "identity")
    vqa_forgetting_level = vqa_task.get("forgetting_level", "object")

    adapter = _create_adapter(config)
    all_classes = dict(sorted(adapter.wnid_to_class.items()))
    if not all_classes:
        raise RuntimeError("No classes found. Check adapter configuration.")

    # Prefer adapter-provided superclass mapping; fall back to JSON file.
    adapter_superclass = getattr(adapter, "wnid_to_superclass", None)
    if adapter_superclass is not None:
        wnid_to_superclass = adapter_superclass
    else:
        sc_path = config.get("paths", {}).get("wnid_to_superclass")
        wnid_to_superclass = _load_json(sc_path) if sc_path else {}

    all_items: list[DatasetItem] = []
    rejects: list[dict] = []
    reject_reason_counter: Counter = Counter()
    class_set: set[str] = set()

    reset_qc_state()
    count = 0

    for sample in adapter.iter_samples():
        if max_samples is not None and count >= max_samples:
            break
        count += 1

        class_set.add(sample.wnid)
        local_seed = _stable_seed_from_path(sample.image_relpath)

        # ---- Prebuilt VQA path (e.g. SpatialMQA) ----
        if sample.prebuilt_qa is not None:
            pqa = sample.prebuilt_qa
            item = DatasetItem(
                id=_build_item_id(sample.image_relpath, sample.wnid, dataset_name),
                image=sample.image_relpath,
                question=pqa.question,
                choices=list(pqa.choices),
                answer_index=pqa.answer_index,
                forgetting_level=pqa.forgetting_level,
                concept_axis=pqa.concept_axis,
                target_split="all",
                meta=_build_meta(sample, wnid_to_superclass),
            )

            passed, reasons = validate_prebuilt_item(item)
            if not passed:
                rejects.append(serialize_reject(item, reasons))
                for r in reasons:
                    reject_reason_counter[r] += 1
                continue

            all_items.append(item)
            continue

        # ---- Standard identity VQA path ----
        hard_pool = _build_hard_pool(sample.wnid, wnid_to_superclass)
        distractors = sample_distractors(
            gt_wnid=sample.wnid,
            hard_pool=hard_pool,
            all_classes=all_classes,
            hard_k=hard_k,
            easy_k=easy_k,
            seed=local_seed,
        )

        if len(distractors) < hard_k + easy_k:
            item = DatasetItem(
                id=_build_item_id(sample.image_relpath, sample.wnid, dataset_name),
                image=sample.image_relpath,
                question=vqa_question,
                choices=[sample.class_name] + distractors,
                answer_index=0,
                forgetting_level=vqa_forgetting_level,
                concept_axis=vqa_concept_axis,
                target_split="all",
                meta=_build_meta(sample, wnid_to_superclass),
            )
            reasons = ["insufficient_unique_distractors"]
            rejects.append(serialize_reject(item, reasons))
            for r in reasons:
                reject_reason_counter[r] += 1
            continue

        choices = _build_choices(sample.class_name, distractors, local_seed)
        answer_index = choices.index(sample.class_name)

        item = DatasetItem(
            id=_build_item_id(sample.image_relpath, sample.wnid, dataset_name),
            image=sample.image_relpath,
            question=vqa_question,
            choices=choices,
            answer_index=answer_index,
            forgetting_level=vqa_forgetting_level,
            concept_axis=vqa_concept_axis,
            target_split="all",
            meta=_build_meta(sample, wnid_to_superclass),
        )

        passed, reasons = validate_item(item)
        if not passed:
            rejects.append(serialize_reject(item, reasons))
            for r in reasons:
                reject_reason_counter[r] += 1
            continue

        all_items.append(item)

    # --- Write JSONL outputs ---
    write_jsonl(output_root / "all.jsonl", all_items, public_schema_only=True)
    (output_root / "internal").mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "internal" / "rejects.jsonl", rejects)

    # --- Compute and write stats ---
    total_accepted = len(all_items)
    total_processed = total_accepted + len(rejects)
    reject_rate = len(rejects) / total_processed if total_processed > 0 else 0.0

    stats = {
        "total_items": total_accepted,
        "total_classes": len(class_set),
        "total_processed": total_processed,
        "total_accepted": total_accepted,
        "total_rejected": len(rejects),
        "reject_rate": round(reject_rate, 4),
        "top_reject_reasons": reject_reason_counter.most_common(5),
    }

    with (output_root / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info("Pipeline complete. Stats: %s", json.dumps(stats, indent=2))
    return stats


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Build identity VQA dataset (ImageNet or COCO)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="vqa_gen/configs/imagenet_identity_mvp.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit total samples processed (for dry runs).",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    stats = run(config, max_samples=args.max_samples)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
