import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    default_weights = str(script_dir / "runs" / "yolov5n_subset" / "weights" / "best.pt")
    default_output = str(script_dir / "runs" / "predictions")

    parser = argparse.ArgumentParser(description="Run YOLOv5n inference and tree counting on drone imagery.")
    parser.add_argument("source", help="Path to input image file or directory of images")
    parser.add_argument("--weights", type=str, default=default_weights,
                        help="Path to trained YOLOv5 model weights")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for tree detections (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Inference image size (default: 416)")
    parser.add_argument("--output-dir", type=str, default=default_output,
                        help="Directory to save annotated images (default: runs/detect/External validation v5 and v7/yolov5_predictions)")
    return parser.parse_args()

def main():
    args = parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] Weights not found at: {weights_path}")
        sys.exit(1)

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"[ERROR] Input source not found: {source_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🌲 YOLOv5n Tree Detection & Counting 🌲")
    print("=" * 60)
    print(f"Weights : {weights_path}")
    print(f"Source  : {source_path}")
    print(f"Conf    : {args.conf}")
    print(f"Output  : {output_dir}")
    print("=" * 60)

    model = YOLO(str(weights_path))

    # Single file or directory
    if source_path.is_file():
        image_files = [source_path]
    else:
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = [p for p in source_path.iterdir() if p.suffix.lower() in valid_exts]

    print(f"[INFO] Processing {len(image_files)} image(s)...")

    for img_file in image_files:
        results = model.predict(
            source=str(img_file),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False
        )
        res = results[0]
        tree_count = len(res.boxes)

        out_name = f"yolov5_annotated_{img_file.name}"
        save_path = output_dir / out_name
        res.save(filename=str(save_path))

        print(f"📸 {img_file.name} -> Detected trees: {tree_count} | Saved to: {save_path.name}")

    print("\n[SUCCESS] All predictions finished. Saved to:", output_dir)

if __name__ == "__main__":
    main()
