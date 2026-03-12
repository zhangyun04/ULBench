import json
from pathlib import Path

from vqa_gen.internal.types import CanonicalSample


class ImageNetAdapter:
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, imagenet_root, wnid_map_path):
        self.imagenet_root = Path(imagenet_root)
        self.wnid_to_class = self._load_wnid_map(wnid_map_path)

    @staticmethod
    def _load_wnid_map(path):
        with Path(path).open("r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise ValueError("wnid_map.json must be a JSON object.")
        return mapping

    def _iter_split_dir(self, split_name):
        split_dir = self.imagenet_root / split_name
        if not split_dir.exists():
            return
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            wnid = class_dir.name
            class_name = self.wnid_to_class.get(wnid)
            if not class_name:
                continue

            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file():
                    continue
                if image_path.suffix.lower() not in self.IMAGE_EXTS:
                    continue
                image_relpath = image_path.relative_to(self.imagenet_root).as_posix()
                yield CanonicalSample(
                    image_relpath=image_relpath,
                    wnid=wnid,
                    class_name=class_name,
                )

    def iter_samples(self):
        for split_name in ("train", "val", "test"):
            for sample in self._iter_split_dir(split_name):
                yield sample
