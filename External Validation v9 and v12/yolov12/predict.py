"""Predict YOLOv12n (see ../README.md for the full protocol). Run from any directory:

    python "External Validation v9 and v12/yolov12/predict.py" --help
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))

from cli import predict_main  # noqa: E402

if __name__ == "__main__":
    predict_main(HERE)
