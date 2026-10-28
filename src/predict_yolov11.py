import argparse
import os
import sys
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Run tree detection and counting using trained YOLO11 model.")
    parser.add_argument("image", help="Path to input image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--weights", type=str, default="runs/detect/tree_detection/yolov11n_phase2/weights/best.pt",
                        help="Path to trained YOLO11 model weights.")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions/yolov11n_predictions",
                        help="Directory to save annotated output images.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not os.path.exists(args.weights):
        print(f"[ERROR] Model weights not found at: '{args.weights}'")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"[ERROR] Input image not found at: '{args.image}'")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 50)
    print("🌲 YOLO11 Tree Detection & Counting 🌲")
    print("=" * 50)
    print(f"Loading weights : {args.weights}")
    model = YOLO(args.weights)

    print(f"Input image     : {args.image}")
    print(f"Confidence      : {args.conf}")

    # Run inference
    results = model(args.image, conf=args.conf)
    result = results[0]

    tree_count = len(result.boxes)
    print("-" * 50)
    print(f"🌲 Total trees detected: {tree_count}")
    print("-" * 50)

    # Save annotated output
    image_basename = os.path.basename(args.image)
    output_path = os.path.join(args.output_dir, f"yolo11_annotated_{image_basename}")
    result.save(filename=output_path)
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    main()
