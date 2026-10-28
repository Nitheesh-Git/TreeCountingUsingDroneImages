import os
import glob
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Evaluate Tree Counting Accuracy on Test Split for YOLOv8n and YOLO11n.")
    parser.add_argument("--v8-weights", type=str, default="runs/detect/tree_detection/baseline_run/weights/best.pt",
                        help="Path to YOLOv8n weights.")
    parser.add_argument("--v11-weights", type=str, default="runs/detect/tree_detection/yolov11n_phase2/weights/best.pt",
                        help="Path to YOLO11n weights.")
    parser.add_argument("--images", type=str, default="datasets/Tree-Top-View/test/images")
    parser.add_argument("--labels", type=str, default="datasets/Tree-Top-View/test/labels")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for counting")
    parser.add_argument("--output-csv", type=str, default="outputs/metrics/counting_comparison.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    print("=" * 70)
    print("🌲 TREE COUNTING EVALUATION: YOLOv8n vs YOLO11n 🌲")
    print("=" * 70)
    print(f"YOLOv8 weights   : {args.v8-weights if hasattr(args, 'v8-weights') else args.v8_weights}")
    print(f"YOLO11 weights   : {args.v11-weights if hasattr(args, 'v11-weights') else args.v11_weights}")
    print(f"Test images      : {args.images}")
    print(f"Confidence       : {args.conf}")
    print("=" * 70)

    # Load models
    v8_model = YOLO(args.v8_weights) if os.path.exists(args.v8_weights) else None
    v11_model = YOLO(args.v11_weights) if os.path.exists(args.v11_weights) else None

    if not v8_model:
        print(f"[ERROR] YOLOv8 weights not found: {args.v8_weights}")
        return
    if not v11_model:
        print(f"[ERROR] YOLO11 weights not found: {args.v11_weights}")
        return

    image_paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    print(f"Evaluating {len(image_paths)} test images...\n")

    results_table = []
    total_gt = 0
    total_v8_pred = 0
    total_v11_pred = 0

    v8_abs_errors = []
    v11_abs_errors = []

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        lbl_path = os.path.join(args.labels, img_name.replace(".jpg", ".txt"))

        # Count ground truth
        gt_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                gt_count = sum(1 for line in f if line.strip())

        # YOLOv8 prediction
        res_v8 = v8_model(img_path, conf=args.conf, verbose=False)[0]
        v8_count = len(res_v8.boxes)

        # YOLO11 prediction
        res_v11 = v11_model(img_path, conf=args.conf, verbose=False)[0]
        v11_count = len(res_v11.boxes)

        v8_err = v8_count - gt_count
        v11_err = v11_count - gt_count

        total_gt += gt_count
        total_v8_pred += v8_count
        total_v11_pred += v11_count

        v8_abs_errors.append(abs(v8_err))
        v11_abs_errors.append(abs(v11_err))

        results_table.append({
            'image': img_name,
            'gt': gt_count,
            'v8_pred': v8_count,
            'v8_abs_err': abs(v8_err),
            'v11_pred': v11_count,
            'v11_abs_err': abs(v11_err)
        })

    v8_mae = sum(v8_abs_errors) / len(v8_abs_errors) if v8_abs_errors else 0.0
    v11_mae = sum(v11_abs_errors) / len(v11_abs_errors) if v11_abs_errors else 0.0

    # Write CSV
    with open(args.output_csv, "w") as f:
        f.write("image,ground_truth,yolov8_pred,yolov8_abs_error,yolo11_pred,yolo11_abs_error\n")
        for r in results_table:
            f.write(f"{r['image']},{r['gt']},{r['v8_pred']},{r['v8_abs_err']},{r['v11_pred']},{r['v11_abs_err']}\n")

    print("=" * 60)
    print(" 🌲 COUNTING EVALUATION SUMMARY (55 TEST IMAGES) 🌲")
    print("=" * 60)
    print(f"Total Ground-Truth Trees        : {total_gt}")
    print(f"Total YOLOv8 Predicted Trees    : {total_v8_pred} (Net Difference: {total_v8_pred - total_gt:+d})")
    print(f"Total YOLO11 Predicted Trees    : {total_v11_pred} (Net Difference: {total_v11_pred - total_gt:+d})")
    print("-" * 60)
    print(f"YOLOv8n Mean Absolute Error (MAE): {v8_mae:.4f} trees/image")
    print(f"YOLO11n Mean Absolute Error (MAE): {v11_mae:.4f} trees/image")
    print("=" * 60)
    print(f"Detailed per-image counting results exported to: {args.output_csv}")

if __name__ == "__main__":
    main()
