"""
Runnable entrypoint for the MVP ImageNet Identity VQA pipeline.

    python vqa_gen/run_build.py --config vqa_gen/configs/imagenet_identity_mvp.yaml
    python vqa_gen/run_build.py --config vqa_gen/configs/imagenet_identity_mvp.yaml --max_samples 200
"""
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from vqa_gen.pipeline.build import main  # noqa: E402

if __name__ == "__main__":
    main()
