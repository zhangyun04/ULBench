import unittest
from pathlib import Path

import yaml

from ulbench.validation import validate_split_registry_data


ROOT = Path(__file__).resolve().parents[1]


class SplitRegistryConsistencyTest(unittest.TestCase):
    def test_encoded_k_and_seed_match_registry_values(self):
        registry_path = ROOT / "scripts" / "split_registry.yaml"
        registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        validate_split_registry_data(registry, check_paths=False)


class ImageNetConfigConsistencyTest(unittest.TestCase):
    def test_readme_imagenet_config_exists_and_is_portable(self):
        config_path = ROOT / "vqa_gen" / "configs" / "imagenet_identity_mvp.yaml"
        self.assertTrue(config_path.is_file())

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["dataset"]["source_type"], "hf_parquet")
        self.assertFalse(Path(config["dataset"]["hf_data_dir"]).is_absolute())
        self.assertEqual(config["paths"]["output_root"],
                         "vqa_gen/output/output_imagenet")
        self.assertEqual(
            config["choices"]["hard_negatives"]
            + config["choices"]["easy_negatives"],
            3,
        )


if __name__ == "__main__":
    unittest.main()
