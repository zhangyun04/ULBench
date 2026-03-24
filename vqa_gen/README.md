# VQA Benchmark & In-Text Unlearning Evaluation

Multi-axis VQA benchmark generation for evaluating Vision-Language Model (VLM) unlearning across diverse forgetting levels and concept axes.

## Supported Datasets

| Dataset | Source Type | Forgetting Level | Concept Axis | # Classes | Source |
|---------|-----------|-----------------|--------------|-----------|--------|
| ImageNet | `imagenet_hf` | object | identity | 1000 | [HuggingFace](https://huggingface.co/datasets/ILSVRC/imagenet-1k) |
| COCO | `coco` | object | identity | 80 | [COCO](https://cocodataset.org/) |
| SpatialMQA | `spatialmqa` | attribute | spatial | — (prebuilt) | [HuggingFace](https://huggingface.co/datasets/liuziyan/SpatialMQA) |
| MIT Indoor-67 | `mit_indoor67` | scene | type_indoor | 67 | [MIT Indoor](https://web.mit.edu/torralba/www/indoor.html) |
| AID | `aid` | scene | type_outdoor | 30 | [AID](https://captain-whu.github.io/AID/) |
| LAD (color) | `lad` | object | attribute_color | 5 domains | [LAD](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset) |
| LAD (shape) | `lad` | object | attribute_shape | 5 domains | [LAD](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset) |
| LAD (size) | `lad` | object | attribute_size | 5 domains | [LAD](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset) |
| LAD (habitat) | `lad` | object | attribute_habitat | 5 domains | [LAD](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset) |
| LAD (behaviour) | `lad` | object | attribute_behaviour | 5 domains | [LAD](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset) |
| Celebrity Faces | `celebrity_faces` | privacy | person | 17 | [Kaggle](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset) |
| Logo-2K+ | `logo2kplus` | privacy | logo | 2341 | [Logo-2K+](https://github.com/msn199959/Logo-2k-plus-Dataset) |

## Structure

```
vqa_gen/
├── configs/                        # Dataset YAML configs
├── adapters/                       # Data loaders per dataset
│   ├── imagenet_hf.py              # ImageNet (HuggingFace)
│   ├── coco.py                     # COCO
│   ├── spatialmqa.py               # SpatialMQA (prebuilt VQA)
│   ├── mit_indoor67.py             # MIT Indoor-67
│   ├── aid.py                      # AID (aerial scenes)
│   ├── lad.py                      # LAD (attribute VQA)
│   ├── celebrity_faces.py          # Celebrity Faces
│   └── logo2kplus.py               # Logo-2K+
├── pipeline/
│   ├── build.py                    # VQA generation pipeline
│   └── qc.py                       # Quality control validation
├── tools/make_explicit_splits.py   # Build experiment splits
├── experiments/intext_unlearning.py # VLM evaluation runner
├── ontology/                       # Category metadata JSONs
│   ├── indoor67_categories.json
│   ├── aid_categories.json
│   └── celebrity_faces.json
└── templates/                      # Question templates
    ├── scene.py
    ├── person.py
    └── logo.py
```

## Data Download & Preparation

### ImageNet / COCO

These use HuggingFace or standard downloads — see original configs.

### SpatialMQA

No manual download needed. The adapter loads directly from HuggingFace:

```yaml
# Uses: liuziyan/SpatialMQA
# Images: reuses COCO images at data/coco
```

Ensure COCO images are available at `data/coco`.

### MIT Indoor-67

1. Download from [MIT Indoor-67](https://web.mit.edu/torralba/www/indoor.html):
   - `indoorCVPR_09.tar` (images)
   - `TrainImages.txt` / `TestImages.txt` (split files)
2. Extract and place so the layout is:
   ```
   data/mit_indoor67/
   ├── Images/
   │   ├── airport_inside/
   │   ├── artstudio/
   │   └── ...
   ├── TrainImages.txt
   └── TestImages.txt
   ```

### AID (Aerial Image Dataset)

1. Download from [AID](https://captain-whu.github.io/AID/) (Google Drive link on the page).
2. Extract and place so the layout is:
   ```
   data/AID/
   ├── Airport/
   ├── BareLand/
   ├── BaseballField/
   └── ... (30 categories)
   ```

### LAD (Large-scale Attribute Dataset)

1. Download from [LAD GitHub](https://github.com/PatrickZH/A-Large-scale-Attribute-Dataset):
   - Images and annotation files
2. Extract and place so the layout is:
   ```
   data/LAD/
   ├── AnimalTrain/ & AnimalTest/
   ├── FruitTrain/ & FruitTest/
   ├── VehicleTrain/ & VehicleTest/
   ├── ElectronicsTrain/ & ElectronicsTest/
   ├── HairstyleTrain/ & HairstyleTest/
   ├── attribute_data/
   │   ├── attribute_list.txt
   │   └── *_attributes_per_class.txt
   └── label/
       └── *.txt (class label files)
   ```

### Celebrity Face Image Dataset

1. Download from [Kaggle](https://www.kaggle.com/datasets/vishesh1412/celebrity-face-image-dataset).
2. Extract and place so the layout is:
   ```
   data/celebrity_faces/Celebrity Faces Dataset/
   ├── Angelina Jolie/
   ├── Brad Pitt/
   ├── Denzel Washington/
   └── ... (17 celebrities)
   ```

### Logo-2K+

1. Download from [Logo-2K+](https://github.com/msn199959/Logo-2k-plus-Dataset) (Google Drive link on the page).
2. Extract and place so the layout is:
   ```
   data/LOGO-2K+/Logo-2K+/
   ├── Clothing/
   │   ├── Adidas/
   │   ├── Nike/
   │   └── ...
   ├── Food/
   └── ... (10 supercategories, 2341 logo classes)
   ```

## Usage

### 1. Generate VQA Dataset

Each dataset has a corresponding config file. Run with:

```bash
python vqa_gen/run_build.py --config <config_path> [--max_samples N]
```

**Original datasets (object identity):**

```bash
# ImageNet
python vqa_gen/run_build.py --config vqa_gen/configs/imagenet_identity_mvp.yaml --max_samples 10000

# COCO
python vqa_gen/run_build.py --config vqa_gen/configs/coco_identity_mvp.yaml --max_samples 10000
```

**Spatial reasoning (attribute-level):**

```bash
# SpatialMQA — prebuilt VQA, downloads from HuggingFace automatically
python vqa_gen/run_build.py --config vqa_gen/configs/spatialmqa_spatial_mvp.yaml --max_samples 5000
```

**Scene classification:**

```bash
# MIT Indoor-67
python vqa_gen/run_build.py --config vqa_gen/configs/mit_indoor67_type_indoor_mvp.yaml --max_samples 5000

# AID (aerial scenes)
python vqa_gen/run_build.py --config vqa_gen/configs/aid_type_outdoor_mvp.yaml --max_samples 5000
```

**Attribute-based VQA (LAD):**

```bash
# Color attributes
python vqa_gen/run_build.py --config vqa_gen/configs/lad_attribute_color_mvp.yaml --max_samples 5000

# Shape attributes
python vqa_gen/run_build.py --config vqa_gen/configs/lad_attribute_shape_mvp.yaml --max_samples 5000

# Size attributes
python vqa_gen/run_build.py --config vqa_gen/configs/lad_attribute_size_mvp.yaml --max_samples 5000

# Habitat attributes
python vqa_gen/run_build.py --config vqa_gen/configs/lad_attribute_habitat_mvp.yaml --max_samples 5000

# Behaviour attributes
python vqa_gen/run_build.py --config vqa_gen/configs/lad_attribute_behaviour_mvp.yaml --max_samples 5000
```

**Privacy-related (person/logo):**

```bash
# Celebrity Faces
python vqa_gen/run_build.py --config vqa_gen/configs/celebrity_faces_person_mvp.yaml --max_samples 1000

# Logo-2K+
python vqa_gen/run_build.py --config vqa_gen/configs/logo2kplus_logo_mvp.yaml --max_samples 5000
```

Each run produces output in the configured `output_root` directory (e.g., `vqa_gen/output/output_logo2kplus/`) containing:
- `train.jsonl` — VQA samples split into `forget`, `retain`, `test` partitions
- Pipeline stats (samples per split, QC pass/reject rate)

### 2. Build Experiment Splits

**Single-target:**

```bash
python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output/output_coco/train.jsonl \
  --forget_class dog \
  --n_forget_test 50 --n_retain_train 200 --n_retain_test 50 \
  --out_dir vqa_gen/experiments/splits/dog/
```

**Multi-target (random K):**

```bash
python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output/output_coco/train.jsonl \
  --k 10 --mode random_k \
  --n_train_per_forget_class 100 --n_test_per_forget_class 100 \
  --n_retain_train 2000 --n_retain_test 500 \
  --seed 123 \
  --out_dir vqa_gen/experiments/splits/randomk10_seed123/
```

**Multi-target (superclass-balanced K):**

```bash
python -m vqa_gen.tools.make_explicit_splits \
  --input_jsonl vqa_gen/output/output_coco/train.jsonl \
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
  --input_jsonl vqa_gen/output/output_coco/train.jsonl \
  --forget_classes_json my_classes.json \
  --n_train_per_forget_class 50 --n_test_per_forget_class 50 \
  --n_retain_train 2000 --n_retain_test 500 \
  --out_dir vqa_gen/experiments/splits/manual5/
```

### 3. Run Evaluation

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

**Quick smoke test:**

```bash
python -m vqa_gen.experiments.intext_unlearning \
  --test_forget_jsonl vqa_gen/experiments/splits/balancedk10_seed123/test_forget.jsonl \
  --test_retain_jsonl vqa_gen/experiments/splits/balancedk10_seed123/test_retain.jsonl \
  --image_root data/coco \
  --max_samples_per_split 10 \
  --out_dir /tmp/smoke_test/
```

## Pipeline Design Notes

- **Standard path**: Adapters yield `CanonicalSample` → pipeline generates question + distractor choices (hard negatives from same supercategory + easy negatives from other categories) → 4-choice VQA.
- **Prebuilt path**: Adapters (SpatialMQA, LAD) yield `CanonicalSample` with `prebuilt_qa` field → pipeline passes through question/choices/answer directly, bypassing distractor sampling.
- **Round-robin iteration**: Datasets without split files (AID, LAD, Celebrity Faces, Logo-2K+) use round-robin across classes to ensure balanced category coverage even with `--max_samples` cap.
- **Class-level splitting**: `forget`/`retain`/`test` splits are assigned at the class level (not sample level), so all images of a given class go to the same split.
- **All questions have exactly 4 choices.**
