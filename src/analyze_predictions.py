import os
import cv2
import glob
import argparse
from ultralytics import YOLO

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    # box format: [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def main():
    parser = argparse.ArgumentParser(description="Quantitative Error Analysis of YOLO Predictions.")
    parser.add_argument("--weights", type=str, default="runs/detect/tree_detection/baseline_run/weights/best.pt")
    parser.add_argument("--images", type=str, default="datasets/Tree-Top-View/valid/images")
    parser.add_argument("--labels", type=str, default="datasets/Tree-Top-View/valid/labels")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions/error_analysis")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for a match")
    parser.add_argument("--max-images", type=int, default=30, help="Max images to process for manual inspection.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.weights}")
    model = YOLO(args.weights)

    image_paths = glob.glob(os.path.join(args.images, "*.jpg"))
    if args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print(f"Processing {len(image_paths)} images with IoU threshold {args.iou_thresh}...")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w, _ = img.shape

        # 1. Load Ground Truth boxes
        gt_boxes = []
        label_path = os.path.join(args.labels, img_name.replace(".jpg", ".txt"))
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = int((x_center - width / 2) * w)
                        y1 = int((y_center - height / 2) * h)
                        x2 = int((x_center + width / 2) * w)
                        y2 = int((y_center + height / 2) * h)
                        gt_boxes.append([x1, y1, x2, y2])

        # 2. Run Inference
        results = model(img_path, conf=args.conf, verbose=False)
        result = results[0]
        
        pred_boxes = []
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            pred_boxes.append({'box': [x1, y1, x2, y2], 'conf': conf})

        # Sort predictions by confidence (highest first) so we match the most confident predictions first
        pred_boxes.sort(key=lambda x: x['conf'], reverse=True)

        # 3. Match Predictions to Ground Truth
        matched_gt_indices = set()
        
        for pred in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt_indices:
                    continue # This ground truth tree has already been found
                    
                iou = calculate_iou(pred['box'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    
            if best_iou >= args.iou_thresh:
                # Match found -> True Positive
                matched_gt_indices.add(best_gt_idx)
                total_tp += 1
                
                # Draw TP (Green predicted box)
                px1, py1, px2, py2 = pred['box']
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
            else:
                # No match found (or duplicate prediction for a matched GT) -> False Positive
                total_fp += 1
                
                # Draw FP (Red predicted box)
                px1, py1, px2, py2 = pred['box']
                cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)

        # 4. Identify False Negatives (unmatched GT boxes)
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx not in matched_gt_indices:
                total_fn += 1
                
                # Draw FN (Blue ground truth box)
                gx1, gy1, gx2, gy2 = gt_box
                cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

        # 5. Add Legend
        cv2.putText(img, "True Positive (Green Pred)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img, "False Positive (Red Pred)", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(img, "False Negative (Blue GT)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Save annotated image
        out_path = os.path.join(args.output_dir, img_name)
        cv2.imwrite(out_path, img)

    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    print("\n" + "=" * 40)
    print("🌲 QUANTITATIVE ERROR ANALYSIS 🌲")
    print("=" * 40)
    print(f"Images Processed: {len(image_paths)}")
    print("-" * 40)
    print(f"True Positives (TP):  {total_tp}")
    print(f"False Positives (FP): {total_fp} (Predictions that were wrong)")
    print(f"False Negatives (FN): {total_fn} (Ground truth trees missed)")
    print("-" * 40)
    print(f"Precision: {precision:.4f}  (TP / (TP + FP))")
    print(f"Recall:    {recall:.4f}  (TP / (TP + FN))")
    print("=" * 40)
    print(f"Categorized annotated images saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
