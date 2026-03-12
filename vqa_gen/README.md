# VQA Benchmark & In-Text Unlearning Evaluation

Identity-based VQA benchmark generation (ImageNet / COCO) with category-level in-text unlearning experiments on Vision-Language Models.

## Structure

```
vqa_gen/
├── configs/                        # Dataset configs (imagenet, coco)
├── adapters/                       # Data loaders (imagenet_hf.py, coco.py)
├── pipeline/build.py               # VQA generation pipeline
├── tools/make_explicit_splits.py   # Build experiment splits
├── experiments/intext_unlearning.py # VLM evaluation runner
└── ontology/                       # ImageNet wnid mappings
```

## Usage

### 1. Generate VQA dataset

```bash
python vqa_gen/run_build.py --config vqa_gen/configs/coco_identity_mvp.yaml --max_samples 10000
```

### 2. Build experiment splits

**Single-target:**

```bash
python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output_coco/train.jsonl \
  --forget_class dog \
  --n_forget_test 50 --n_retain_train 200 --n_retain_test 50 \
  --out_dir vqa_gen/experiments/splits/dog/
```

**Multi-target (random K):**

```bash
python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output_coco/train.jsonl \
  --k 10 --mode random_k \
  --n_train_per_forget_class 100 --n_test_per_forget_class 100 \
  --n_retain_train 2000 --n_retain_test 500 \
  --seed 123 \
  --out_dir vqa_gen/experiments/splits/randomk10_seed123/
```

**Multi-target (superclass-balanced K):**

```bash
python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output_coco/train.jsonl \
  --k 10 --mode superclass_balanced_k \
  --n_train_per_forget_class 50 --n_test_per_forget_class 50 \
  --n_retain_train 1000 --n_retain_test 500 \
  --seed 123 \
  --out_dir vqa_gen/experiments/splits/balancedk10_seed123/
```

**Multi-target (explicit class list):**

```bash
echo '["dog","cat","horse","bus","pizza"]' > my_classes.json

python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output_coco/train.jsonl \
  --forget_classes_json my_classes.json \
  --n_train_per_forget_class 50 --n_test_per_forget_class 50 \
  --n_retain_train 2000 --n_retain_test 500 \
  --out_dir vqa_gen/experiments/splits/manual5/
```

### 3. Run evaluation

**Single-target (baseline only):**

```bash
python -m vqa_gen.experiments.intext_unlearning \
  --test_forget_jsonl vqa_gen/experiments/splits/dog/test_forget.jsonl \
  --test_retain_jsonl vqa_gen/experiments/splits/dog/test_retain.jsonl \
  --forget_class dog \
  --image_root data/coco \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --out_dir vqa_gen/experiments/results/dog/
```

**Multi-target with oracle:**

```bash
python -m vqa_gen.experiments.intext_unlearning \
  --test_forget_jsonl vqa_gen/experiments/splits/balancedk10_seed123/test_forget.jsonl \
  --test_retain_jsonl vqa_gen/experiments/splits/balancedk10_seed123/test_retain.jsonl \
  --forget_classes_json vqa_gen/experiments/splits/balancedk10_seed123/forget_classes.json \
  --image_root data/coco \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --run_oracle_hard \
  --out_dir vqa_gen/experiments/results/balancedk10_seed123/
```

**Multi-target (auto-inferred classes):**

```bash
python -m vqa_gen.experiments.intext_unlearning \
  --test_forget_jsonl vqa_gen/experiments/splits/randomk10_seed123/test_forget.jsonl \
  --test_retain_jsonl vqa_gen/experiments/splits/randomk10_seed123/test_retain.jsonl \
  --image_root data/coco \
  --model_name Qwen/Qwen3-VL-4B-Instruct \
  --out_dir vqa_gen/experiments/results/randomk10_baseline/
```

**Quick smoke test:**

```bash
python -m vqa_gen.experiments.intext_unlearning \
  --test_forget_jsonl vqa_gen/experiments/splits/balancedk10_seed123/test_forget.jsonl \
  --test_retain_jsonl vqa_gen/experiments/splits/balancedk10_seed123/test_retain.jsonl \
  --image_root data/coco \
  --max_samples_per_split 10 \
  --out_dir /tmp/smoke_test/
```
