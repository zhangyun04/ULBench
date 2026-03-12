import hashlib
import io
import logging
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample

logger = logging.getLogger(__name__)


def _clean_class_name(raw_name: str) -> str:
    """Extract first name from comma-separated synset description.

    HF ImageNet labels look like "golden retriever" or
    "great white shark, white shark, man-eater, ...".
    We take only the first part to keep choice text short and QC-safe.
    """
    return raw_name.split(",")[0].strip()


def _safe_dirname(name: str) -> str:
    """Make a class name safe for use as a directory component."""
    return (
        name
        .replace(" ", "_")
        .replace("/", "_")
        .replace("'", "")
        .replace("(", "")
        .replace(")", "")
    )


class ImageNetHFAdapter:
    """Adapter for HuggingFace parquet-sharded ImageNet-1k."""

    def __init__(self, hf_data_dir: str, splits=None):
        from datasets import load_dataset

        self.hf_data_dir = Path(hf_data_dir)
        self.cache_root = self.hf_data_dir.parent / "cache_images"
        self.splits = splits or ["train", "validation", "test"]

        self._label_names: list[str] = []
        self._label_to_clean: dict[int, str] = {}
        self.wnid_to_class: dict[str, str] = {}

        self._init_label_names(load_dataset)

    def _init_label_names(self, load_dataset_fn):
        for split_name in self.splits:
            try:
                ds = load_dataset_fn(
                    "parquet",
                    data_dir=str(self.hf_data_dir),
                    split=split_name,
                )
                self._label_names = ds.features["label"].names
                del ds
                break
            except Exception:
                continue

        if not self._label_names:
            raise RuntimeError(
                f"Could not load label names from any split in {self.hf_data_dir}. "
                f"Tried splits: {self.splits}"
            )

        for idx, raw_name in enumerate(self._label_names):
            clean = _clean_class_name(raw_name)
            self._label_to_clean[idx] = clean
            self.wnid_to_class[clean] = clean

        logger.info("Loaded %d classes from HF parquet dataset", len(self.wnid_to_class))

    def _cache_image(self, image, split_name: str, class_name: str, label_id: int) -> str:
        """Save image to deterministic cache path and return relpath.

        Returns path relative to ``cache_root`` so that the first component
        is the split name (needed by ``_build_item_id``).
        """
        image = image.convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        h = hashlib.sha1(img_bytes).hexdigest()[:8]
        safe_cls = _safe_dirname(class_name)
        filename = f"{split_name}_{label_id}_{h}.jpg"

        cache_dir = self.cache_root / split_name / safe_cls
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / filename

        if not cache_path.exists():
            cache_path.write_bytes(img_bytes)

        return f"{split_name}/{safe_cls}/{filename}"

    def iter_samples(self):
        from datasets import load_dataset

        for split_name in self.splits:
            try:
                ds = load_dataset(
                    "parquet",
                    data_dir=str(self.hf_data_dir),
                    split=split_name,
                )
            except Exception as exc:
                logger.warning("Skipping split %s: %s", split_name, exc)
                continue

            for example in ds:
                label_id: int = example["label"]
                class_name = self._label_to_clean[label_id]

                relpath = self._cache_image(
                    example["image"], split_name, class_name, label_id,
                )

                yield CanonicalSample(
                    image_relpath=relpath,
                    wnid=class_name,
                    class_name=class_name,
                )

            del ds
