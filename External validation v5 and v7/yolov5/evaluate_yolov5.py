import argparse
import glob
import json
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    default_weights = str(script_dir / "runs" / "yolov5n_subset" / "weights" / "best.pt")
    default_data = str(repo_root / "External validation v5 and v7" / "data_subset_595.yaml")
    default_project = str(script_dir / "runs" / "evaluation")

    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv5n on validation or test dataset.")
    parser.add_argument("--weights", type=str, default=default_weights,
                        help="Path to trained YOLOv5 model weights (.pt)")
    parser.add_argument("--data", type=str, default=default_data,
                        help="Path to data_subset_595.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"],
                        help="Dataset split to evaluate ('val' or 'test')")
    parser.add_argument("--project", type=str, default=default_project,
                        help="Project directory to save evaluation artifacts")
    parser.add_argument("--name", type=str, default="yolov5n_test_eval",
                        help="Evaluation run directory name")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--eval-counting", action="store_true", default=True,
                        help="Also evaluate per-image tree counting error (MAE) against label files")
    return parser.parse_args()

def evaluate_counting_mae(model, images_dir, labels_dir, conf=0.25):
    """Compute Mean Absolute Error (MAE) for tree counting."""
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not img_paths:
        return None, None, None

    total_gt = 0
    total_pred = 0
    abs_errors = []

    for p in img_paths:
        basename = os.path.basename(p)
        lbl_path = os.path.join(labels_dir, Path(basename).stem + ".txt")

        gt_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                gt_count = sum(1 for line in f if line.strip())

        results = model.predict(p, conf=conf, verbose=False)
        pred_count = len(results[0].boxes)

        total_gt += gt_count
        total_pred += pred_count
        abs_errors.append(abs(pred_count - gt_count))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return mae, total_gt, total_pred

def main():
    args = parse_args()
    args.project = str(Path(args.project).resolve())

    print("=" * 70)
    print("🌲 YOLOv5n Model Evaluation & Metrics Report 🌲")
    print("=" * 70)
    print(f"Weights path     : {args.weights}")
    print(f"Dataset config   : {args.data}")
    print(f"Evaluation split : {args.split}")
    print(f"Image size       : {args.imgsz}")
    print(f"Batch size       : {args.batch}")
    print(f"Device           : {args.device}")
    print(f"Output directory : {Path(args.project) / args.name}")
    print("=" * 70)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"[ERROR] Dataset configuration not found: {data_path}")
        sys.exit(1)

    print(f"\n[INFO] Loading YOLOv5n weights from: {weights_path}")
    model = YOLO(str(weights_path))

    # Standard YOLO validation
    print(f"[INFO] Running evaluation on '{args.split}' split...")
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=True
    )

    prec = float(metrics.box.mp)
    rec = float(metrics.box.mr)
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    speed = metrics.speed or {}
    inf_latency = speed.get("inference", 0.0)
    pre_latency = speed.get("preprocess", 0.0)
    post_latency = speed.get("postprocess", 0.0)
    total_latency = pre_latency + inf_latency + post_latency

    print("\n" + "=" * 70)
    print(f"📊 YOLOv5n EVALUATION RESULTS ({args.split.upper()} SPLIT) 📊")
    print("=" * 70)
    print(f"Precision (P)    : {prec:.4f}")
    print(f"Recall (R)       : {rec:.4f}")
    print(f"mAP@0.50         : {map50:.4f}")
    print(f"mAP@0.50:0.95    : {map50_95:.4f}")
    print(f"Preprocess       : {pre_latency:.2f} ms/image")
    print(f"Inference Latency: {inf_latency:.2f} ms/image")
    print(f"Postprocess      : {post_latency:.2f} ms/image")
    print(f"Total Latency    : {total_latency:.2f} ms/image")
    print("=" * 70)

    counting_mae = None
    if args.eval_counting:
        repo_root = Path(__file__).resolve().parents[2]
        split_name = "test" if args.split == "test" else "valid"
        images_dir = repo_root / "datasets" / "Tree-Top-View" / split_name / "images"
        labels_dir = repo_root / "datasets" / "Tree-Top-View" / split_name / "labels"

        if images_dir.exists() and labels_dir.exists():
            print(f"\n[INFO] Evaluating tree-counting MAE on {split_name} split...")
            counting_mae, total_gt, total_pred = evaluate_counting_mae(model, str(images_dir), str(labels_dir), conf=0.25)
            print(f"Total Ground-Truth Trees : {total_gt}")
            print(f"Total Predicted Trees    : {total_pred}")
            print(f"Tree Counting MAE        : {counting_mae:.3f} trees/image")
            print("=" * 70)

    # Save metrics JSON
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": "YOLOv5n",
        "weights": str(weights_path),
        "split": args.split,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "map50": round(map50, 4),
        "map50_95": round(map50_95, 4),
        "inference_ms": round(inf_latency, 2),
        "preprocess_ms": round(pre_latency, 2),
        "postprocess_ms": round(post_latency, 2),
        "total_ms": round(total_latency, 2),
        "counting_mae": round(counting_mae, 3) if counting_mae is not None else None,
        "total_gt_trees": int(total_gt) if counting_mae is not None else None,
        "total_pred_trees": int(total_pred) if counting_mae is not None else None
    }

    summary_file = out_dir / "metrics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to: {summary_file}")

if __name__ == "__main__":
    main()
