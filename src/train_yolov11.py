import os
import sys
import time
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLO11n on Tree-Top-View dataset (Phase 2).")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Pretrained model weights or config")
    parser.add_argument("--data", type=str, default="datasets/Tree-Top-View/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
    parser.add_argument("--project", type=str, default="tree_detection", help="Project save directory")
    parser.add_argument("--name", type=str, default="yolov11n_phase2", help="Run name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("🌲 YOLO11 Tree Detection - Phase 2 Training 🌲")
    print("=" * 70)
    print(f"Model weights     : {args.model}")
    print(f"Dataset config    : {args.data}")
    print(f"Image Size        : {args.imgsz}x{args.imgsz}")
    print(f"Batch Size        : {args.batch}")
    print(f"Maximum Epochs    : {args.epochs}")
    print(f"Patience          : {args.patience}")
    print(f"Device            : {args.device}")
    print(f"Random Seed       : {args.seed}")
    print(f"Output Project    : {args.project}")
    print(f"Run Name          : {args.name}")
    print("=" * 70)

    # 1. Verify dataset configuration exists
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Dataset configuration not found at: {args.data}")

    # 2. Verify model weights exist
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found at: {args.model}")

    # 3. Load YOLO11 model
    print(f"\n[INFO] Loading {args.model}...")
    model = YOLO(args.model)
    print(f"[SUCCESS] Model loaded successfully. Task: {model.task}")

    # Output directory path
    output_dir = os.path.join("runs", "detect", args.project, args.name)
    print(f"[INFO] Training results will be saved to: {output_dir}")

    start_time = time.time()
    print("\n[INFO] Starting YOLO11n training loop...\n")

    # 4. Train the model
    # Note: Ultralytics handles early stopping, checkpoints, validation, and metrics logging automatically.
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=args.project,
        name=args.name,
        seed=args.seed,
        workers=args.workers,
        exist_ok=True,
        verbose=True,
        val=True,
        save=True,
        plots=True
    )

    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    print("\n" + "=" * 70)
    print("🎉 YOLO11n Training Completed 🎉")
    print("=" * 70)
    print(f"Total training duration : {int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s ({elapsed_time:.1f}s)")
    print(f"Output directory        : {output_dir}/")
    print(f"Best model weights      : {os.path.join(output_dir, 'weights', 'best.pt')}")
    print(f"Last model weights      : {os.path.join(output_dir, 'weights', 'last.pt')}")
    print("=" * 70)

if __name__ == "__main__":
    main()
