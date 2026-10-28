import argparse
import os
import sys
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO11 model on validation or test dataset.")
    parser.add_argument("--weights", type=str, default="runs/detect/tree_detection/yolov11n_phase2/weights/best.pt",
                        help="Path to trained YOLO11 model weights.")
    parser.add_argument("--data", type=str, default="datasets/Tree-Top-View/data.yaml",
                        help="Path to dataset configuration file.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Dataset split to evaluate (val or test).")
    parser.add_argument("--project", type=str, default="outputs/metrics",
                        help="Project directory to save results.")
    parser.add_argument("--name", type=str, default="yolov11n_test_evaluation",
                        help="Name of evaluation run folder.")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda).")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 60)
    print("🌲 YOLO11 Model Evaluation 🌲")
    print("=" * 60)
    print(f"Weights path     : {args.weights}")
    print(f"Dataset config   : {args.data}")
    print(f"Target split     : {args.split}")
    print(f"Image size       : {args.imgsz}")
    print(f"Output directory : {os.path.join(args.project, args.name)}")
    print("=" * 60)

    # 1. Verify weights exist
    if not os.path.exists(args.weights):
        print(f"[ERROR] Model weights not found at: '{args.weights}'")
        sys.exit(1)

    # 2. Verify dataset config exists
    if not os.path.exists(args.data):
        print(f"[ERROR] Dataset configuration not found at: '{args.data}'")
        sys.exit(1)

    # 3. Load trained model
    print(f"\n[INFO] Loading YOLO11 model from {args.weights}...")
    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    # 4. Run validation on the specified split
    print(f"[INFO] Running evaluation on '{args.split}' split...")
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True
    )

    # 5. Extract and print metrics cleanly
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    speed = metrics.speed  # dictionary with preprocess, inference, loss, postprocess in ms

    print("\n" + "=" * 60)
    print(f" 📊 YOLO11 EVALUATION METRICS ({args.split.upper()} SPLIT) 📊")
    print("=" * 60)
    print(f"Precision          : {precision:.4f}")
    print(f"Recall             : {recall:.4f}")
    print(f"mAP@0.5            : {map50:.4f}")
    print(f"mAP@0.5:0.95       : {map50_95:.4f}")
    if speed:
        inference_ms = speed.get('inference', 0.0)
        preprocess_ms = speed.get('preprocess', 0.0)
        postprocess_ms = speed.get('postprocess', 0.0)
        total_ms = preprocess_ms + inference_ms + postprocess_ms
        print(f"Inference latency  : {inference_ms:.2f} ms/img (preprocess: {preprocess_ms:.2f}ms, postprocess: {postprocess_ms:.2f}ms)")
        print(f"Total per-image    : {total_ms:.2f} ms/img")
    print("=" * 60)

    output_dir = os.path.join(args.project, args.name)
    print(f"[INFO] Evaluation artifacts (confusion matrix, PR curve, etc.) saved to:\n  {output_dir}")

if __name__ == "__main__":
    main()
