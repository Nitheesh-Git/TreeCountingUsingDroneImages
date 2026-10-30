import argparse
import json
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    # Resolve project root relative to this script
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    default_data = str(repo_root / "External validation v5 and v7" / "data_subset_595.yaml")
    default_model = str(repo_root / "yolov5nu.pt") if (repo_root / "yolov5nu.pt").exists() else "yolov5nu.pt"
    default_project = str(script_dir / "runs")

    parser = argparse.ArgumentParser(description="Train YOLOv5n on the 595-image reproducible tree dataset subset.")
    parser.add_argument("--model", type=str, default=default_model,
                        help="Pretrained model weights or config (default: yolov5nu.pt)")
    parser.add_argument("--data", type=str, default=default_data,
                        help="Path to data_subset_595.yaml")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Maximum training epochs (default: 30)")
    parser.add_argument("--imgsz", type=int, default=416,
                        help="Input image size (default: 416)")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8, optimized for available CPU RAM)")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience in epochs (default: 5)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for training (default: cpu)")
    parser.add_argument("--project", type=str, default=default_project,
                        help="Directory to save runs (default: External validation v5 and v7/yolov5/runs)")
    parser.add_argument("--name", type=str, default="yolov5n_subset",
                        help="Run name (default: yolov5n_subset)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    return parser.parse_args()

def main():
    args = parse_args()
    args.project = str(Path(args.project).resolve())

    print("=" * 75)
    print("🌲 YOLOv5n Comparison Model Training (595-Image Subset) 🌲")
    print("=" * 75)
    print(f"Pretrained Model  : {args.model}")
    print(f"Dataset Config    : {args.data}")
    print(f"Image Size        : {args.imgsz}x{args.imgsz}")
    print(f"Batch Size        : {args.batch}")
    print(f"Max Epochs        : {args.epochs}")
    print(f"Patience          : {args.patience}")
    print(f"Device            : {args.device}")
    print(f"Random Seed       : {args.seed}")
    print(f"Output Project    : {args.project}")
    print(f"Run Name          : {args.name}")
    print("=" * 75)

    # 1. Verify dataset config
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        print(f"[ERROR] Dataset configuration not found at: {data_path}")
        sys.exit(1)

    # 2. Load model
    print(f"\n[INFO] Initializing YOLOv5n model from: {args.model}...")
    try:
        model = YOLO(args.model)
        print(f"[SUCCESS] Model initialized successfully. Task: {model.task}")
    except Exception as e:
        print(f"[ERROR] Failed to load model '{args.model}': {e}")
        sys.exit(1)

    target_run_dir = Path(args.project) / args.name
    print(f"[INFO] Destination run directory: {target_run_dir}")

    start_time = time.time()
    print("\n[INFO] Starting YOLOv5n training loop on CPU...")
    print(f"[NOTE] Training on 595-image subset; validation split has 116 images.\n")

    # 3. Train
    try:
        train_results = model.train(
            data=str(data_path),
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
    except Exception as e:
        print(f"\n[ERROR] Training interrupted or failed: {e}")
        sys.exit(1)

    elapsed_seconds = time.time() - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_duration = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"

    print("\n" + "=" * 75)
    print("🎉 YOLOv5n Training Finished 🎉")
    print("=" * 75)
    print(f"Total Duration    : {formatted_duration} ({elapsed_seconds:.1f}s)")
    print(f"Run Directory     : {target_run_dir}")

    best_pt = target_run_dir / "weights" / "best.pt"
    last_pt = target_run_dir / "weights" / "last.pt"
    print(f"Best Weights      : {best_pt} (exists: {best_pt.exists()})")
    print(f"Last Weights      : {last_pt} (exists: {last_pt.exists()})")

    # Record training metadata summary
    summary_data = {
        "model_name": "YOLOv5n",
        "pretrained_weights": str(args.model),
        "dataset_config": str(data_path),
        "subset_image_count": 595,
        "validation_image_count": 116,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "formatted_duration": formatted_duration,
        "best_weights_path": str(best_pt) if best_pt.exists() else None,
        "last_weights_path": str(last_pt) if last_pt.exists() else None
    }

    summary_file = target_run_dir / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON      : {summary_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
