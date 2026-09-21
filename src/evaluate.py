import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 model on the test dataset.")
    parser.add_argument("--weights", type=str, default="runs/detect/tree_detection/baseline_run/weights/best.pt", 
                        help="Path to the trained model weights.")
    parser.add_argument("--data", type=str, default="datasets/Tree-Top-View/data.yaml", 
                        help="Path to the dataset configuration file.")
    parser.add_argument("--project", type=str, default="outputs/metrics", 
                        help="Project directory to save results.")
    parser.add_argument("--name", type=str, default="test_evaluation", 
                        help="Name of the evaluation run (creates a folder).")
    args = parser.parse_args()

    # 1. Load the trained model
    print(f"Loading model weights from: {args.weights}")
    try:
        model = YOLO(args.weights)
    except FileNotFoundError:
        print(f"Error: Model weights not found at '{args.weights}'.")
        return

    # 2. Evaluate on the 'test' split specified in data.yaml
    print(f"Evaluating model on test split defined in: {args.data}")
    # Ultralytics val() function automatically generates metrics, confusion matrix, and plots.
    # split="test" explicitly targets the test/ folder instead of valid/
    metrics = model.val(
        data=args.data,
        split="test",
        project=args.project,
        name=args.name,
        exist_ok=True
    )
    
    # 3. Print the metrics cleanly
    print("\n" + "=" * 40)
    print("🌲 TEST EVALUATION METRICS 🌲")
    print("=" * 40)
    
    # Extract metrics from the YOLO metrics object
    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP50:     {map50:.4f}")
    print(f"mAP50-95:  {map50_95:.4f}")
    print("=" * 40)
    
    # 4, 5 & 6. The val() function automatically saves plots and confusion matrices
    output_dir = os.path.join(args.project, args.name)
    print(f"Evaluation artifacts (including confusion matrix and plots) are saved in: {output_dir}")

if __name__ == "__main__":
    main()
