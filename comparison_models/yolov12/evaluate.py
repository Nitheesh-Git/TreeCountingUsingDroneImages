"""Evaluate YOLOv12n (see ../README.md for the full protocol). Run from any directory:

    python comparison_models/yolov12/evaluate.py --help
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

from cli import evaluate_main  # noqa: E402

if __name__ == "__main__":
    evaluate_main(HERE)
