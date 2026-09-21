import argparse
import os
from ultralytics import YOLO

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run tree detection on an image using a trained YOLOv8 model.")
    parser.add_argument("image", help="Path to the input image.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections (default: 0.25).")
    parser.add_argument("--weights", type=str, default="runs/detect/tree_detection/baseline_run/weights/best.pt", 
                        help="Path to the trained model weights.")
    parser.add_argument("--output-dir", type=str, default="outputs/predictions", 
                        help="Directory to save the annotated output images.")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the trained model
    print(f"Loading model weights from: {args.weights}")
    try:
        model = YOLO(args.weights)
    except FileNotFoundError:
        print(f"Error: Model weights not found at '{args.weights}'. Please check the path.")
        return

    # 2 & 3. Run inference with the specified confidence threshold
    print(f"Running detection on: {args.image} (Confidence threshold: {args.conf})")
    
    # model() returns a list of Result objects; we get the first one since we passed a single image
    results = model(args.image, conf=args.conf)
    result = results[0]
    
    # 6. Count the total number of detected trees (each box is one detection)
    tree_count = len(result.boxes)
    
    # 7. Print the total count in the terminal
    print("-" * 40)
    print(f"🌲 Total trees detected: {tree_count}")
    print("-" * 40)

    # 4, 5 & 8. Draw bounding boxes, confidences, and save the annotated image
    # The result.save() function automatically draws boxes, labels, and confidences
    image_basename = os.path.basename(args.image)
    output_path = os.path.join(args.output_dir, f"annotated_{image_basename}")
    
    result.save(filename=output_path)
    
    print(f"Annotated image saved to: {output_path}")

if __name__ == "__main__":
    main()
