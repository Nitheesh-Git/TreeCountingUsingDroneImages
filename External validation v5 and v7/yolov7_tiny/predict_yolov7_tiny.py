import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
YOLOV7_REPO = SCRIPT_DIR / "yolov7_repo"
SHIMS_DIR = SCRIPT_DIR / "shims"

sys.path.insert(0, str(SHIMS_DIR))
sys.path.insert(0, str(YOLOV7_REPO))

import cv2
import numpy as np
import torch
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device

def parse_args():
    default_weights = str(SCRIPT_DIR / "runs" / "yolov7_tiny_subset" / "weights" / "best.pt")
    default_output = str(SCRIPT_DIR / "runs" / "predictions")

    parser = argparse.ArgumentParser(description="Run YOLOv7-tiny inference and tree counting.")
    parser.add_argument("source", help="Path to input image file or directory")
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--output-dir", type=str, default=default_output, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    return parser.parse_args()

def main():
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] Weights not found: {weights_path}")
        sys.exit(1)

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"[ERROR] Source not found: {source_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🌲 YOLOv7-tiny Tree Detection & Counting 🌲")
    print("=" * 60)
    print(f"Weights : {weights_path}")
    print(f"Source  : {source_path}")
    print(f"Conf    : {args.conf}")
    print(f"Output  : {out_dir}")
    print("=" * 60)

    device = select_device(args.device)
    model = attempt_load(str(weights_path), map_location=device).float()
    model.eval()

    if source_path.is_file():
        image_files = [source_path]
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = [p for p in source_path.iterdir() if p.suffix.lower() in valid_exts]

    print(f"[INFO] Processing {len(image_files)} image(s)...")

    for img_file in image_files:
        img0 = cv2.imread(str(img_file))
        if img0 is None:
            continue
        img = letterbox(img0, args.imgsz, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=args.conf, iou_thres=args.iou)[0]

        tree_count = 0
        if pred is not None and len(pred):
            pred[:, :4] = scale_coords(img.shape[1:], pred[:, :4], img0.shape).round()
            tree_count = len(pred)
            for *xyxy, conf, cls in reversed(pred):
                label = f"tree-top {conf:.2f}"
                plot_one_box(xyxy, img0, label=label, color=(34, 139, 34), line_thickness=2)

        out_path = out_dir / f"yolov7_annotated_{img_file.name}"
        cv2.imwrite(str(out_path), img0)
        print(f"📸 {img_file.name} -> Detected trees: {tree_count} | Saved to: {out_path.name}")

    print("\n[SUCCESS] All predictions finished. Saved to:", out_dir)

if __name__ == "__main__":
    main()
