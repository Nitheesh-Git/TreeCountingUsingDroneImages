import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
YOLOV7_REPO = SCRIPT_DIR / "yolov7_repo"
SHIMS_DIR = SCRIPT_DIR / "shims"

sys.path.insert(0, str(SHIMS_DIR))
sys.path.insert(0, str(YOLOV7_REPO))

import torch
import yaml
from models.experimental import attempt_load
from utils.general import check_file, set_logging, increment_path
from utils.torch_utils import select_device
import train as yolov7_train

def parse_args():
    default_weights = str(SCRIPT_DIR / "yolov7-tiny.pt")
    default_cfg = str(YOLOV7_REPO / "cfg" / "training" / "yolov7-tiny.yaml")
    default_data = str(SCRIPT_DIR / "data_subset_595_yolov7.yaml")
    default_hyp = str(YOLOV7_REPO / "data" / "hyp.scratch.tiny.yaml")
    default_project = str(SCRIPT_DIR / "runs")

    parser = argparse.ArgumentParser(description="Train YOLOv7-tiny on 595-image reproducible tree dataset subset.")
    parser.add_argument("--weights", type=str, default=default_weights, help="Initial weights path")
    parser.add_argument("--cfg", type=str, default=default_cfg, help="Model yaml path")
    parser.add_argument("--data", type=str, default=default_data, help="Dataset yaml path")
    parser.add_argument("--hyp", type=str, default=default_hyp, help="Hyperparameters path")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--img-size", nargs="+", type=int, default=[416, 416], help="[train, test] image sizes")
    parser.add_argument("--device", default="cpu", help="Device to use for training (default: cpu)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--project", default=default_project, help="Save project directory")
    parser.add_argument("--name", default="yolov7_tiny_subset", help="Save run name")
    parser.add_argument("--exist-ok", action="store_true", default=True, help="Existing project/name ok")
    parser.add_argument("--rect", action="store_true", default=False, help="Rectangular training")
    parser.add_argument("--single-cls", action="store_true", default=False, help="Train as single-class")
    parser.add_argument("--adam", action="store_true", default=False, help="Use Adam optimizer")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 75)
    print("🌲 YOLOv7-tiny Comparison Model Training (595-Image Subset) 🌲")
    print("=" * 75)
    print(f"Pretrained Model  : {args.weights}")
    print(f"Model Config      : {args.cfg}")
    print(f"Dataset Config    : {args.data}")
    print(f"Hyperparameters   : {args.hyp}")
    print(f"Image Size        : {args.img_size}")
    print(f"Batch Size        : {args.batch_size}")
    print(f"Max Epochs        : {args.epochs}")
    print(f"Patience          : {args.patience}")
    print(f"Device            : {args.device}")
    print(f"Random Seed       : {args.seed}")
    print(f"Output Project    : {args.project}")
    print(f"Run Name          : {args.name}")
    print("=" * 75)

    # Populate opt namespace expected by yolov7 train.py
    opt = argparse.Namespace()
    opt.weights = args.weights
    opt.cfg = args.cfg
    opt.data = args.data
    opt.hyp = args.hyp
    opt.epochs = args.epochs
    opt.batch_size = args.batch_size
    opt.total_batch_size = args.batch_size
    opt.img_size = args.img_size if len(args.img_size) == 2 else [args.img_size[0], args.img_size[0]]
    opt.rect = args.rect
    opt.resume = False
    opt.nosave = False
    opt.notest = False
    opt.noautoanchor = True
    opt.evolve = False
    opt.bucket = ""
    opt.cache_images = False
    opt.image_weights = False
    opt.device = args.device
    opt.multi_scale = False
    opt.single_cls = args.single_cls
    opt.adam = args.adam
    opt.sync_bn = False
    opt.local_rank = -1
    opt.global_rank = -1
    opt.world_size = 1
    opt.workers = args.workers
    opt.project = str(Path(args.project).resolve())
    opt.name = args.name
    opt.exist_ok = args.exist_ok
    opt.quad = False
    opt.linear_lr = False
    opt.label_smoothing = 0.0
    opt.upload_dataset = False
    opt.bbox_interval = -1
    opt.save_period = -1
    opt.artifact_alias = "latest"
    opt.freeze = [0]
    opt.v5_metric = True
    opt.patience = args.patience
    opt.seed = args.seed

    set_logging(opt.global_rank)
    opt.save_dir = str(Path(opt.project) / opt.name)
    Path(opt.save_dir).mkdir(parents=True, exist_ok=True)

    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)

    device = select_device(opt.device, batch_size=opt.batch_size)

    start_time = time.time()
    print("\n[INFO] Starting YOLOv7-tiny training on CPU...")

    # Run training
    results = yolov7_train.train(hyp, opt, device, tb_writer=None)

    elapsed_seconds = time.time() - start_time
    hours, rem = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    formatted_duration = f"{int(hours):02d}h {int(minutes):02d}m {int(seconds):02d}s"

    save_dir_path = Path(opt.save_dir)
    best_pt = save_dir_path / "weights" / "best.pt"
    last_pt = save_dir_path / "weights" / "last.pt"

    print("\n" + "=" * 75)
    print("🎉 YOLOv7-tiny Training Finished 🎉")
    print("=" * 75)
    print(f"Total Duration    : {formatted_duration} ({elapsed_seconds:.1f}s)")
    print(f"Run Directory     : {save_dir_path}")
    print(f"Best Weights      : {best_pt} (exists: {best_pt.exists()})")
    print(f"Last Weights      : {last_pt} (exists: {last_pt.exists()})")

    summary_data = {
        "model_name": "YOLOv7-tiny",
        "pretrained_weights": str(args.weights),
        "model_cfg": str(args.cfg),
        "dataset_config": str(args.data),
        "subset_image_count": 595,
        "validation_image_count": 116,
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "device": args.device,
        "seed": args.seed,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "formatted_duration": formatted_duration,
        "best_weights_path": str(best_pt) if best_pt.exists() else None,
        "last_weights_path": str(last_pt) if last_pt.exists() else None
    }

    summary_file = save_dir_path / "training_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"Summary JSON      : {summary_file}")
    print("=" * 75)

if __name__ == "__main__":
    main()
