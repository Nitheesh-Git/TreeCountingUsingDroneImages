"""Command-line entry points shared by the per-model train/evaluate/predict wrappers."""

import argparse

from pipeline import evaluate, load_config, predict, train


def train_main(model_dir):
    cfg = load_config(model_dir)
    t = cfg["train"]
    p = argparse.ArgumentParser(description=f"Train {cfg['model_name']} on the half-size subset.")
    p.add_argument("--epochs", type=int, help=f"Maximum epochs (config: {t['epochs']})")
    p.add_argument("--patience", type=int, help=f"Early-stopping patience (config: {t['patience']})")
    p.add_argument("--batch", type=int, help=f"Batch size (config: {t['batch']})")
    p.add_argument("--imgsz", type=int, help=f"Image size (config: {t['imgsz']})")
    p.add_argument("--device", type=str, help=f"cpu or a CUDA index (config: {t['device']})")
    p.add_argument("--workers", type=int, help=f"DataLoader workers (config: {t['workers']})")
    p.add_argument("--time", type=float,
                   help="Training time budget in hours; Ultralytics fits the epoch count and LR schedule to it")
    p.add_argument("--run-name", type=str, help=f"Run folder name (config: {cfg['run_name']})")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run from last.pt")
    a = p.parse_args()
    if a.run_name:
        cfg["run_name"] = a.run_name
    train(cfg, {"epochs": a.epochs, "patience": a.patience, "batch": a.batch, "imgsz": a.imgsz,
                "device": a.device, "workers": a.workers, "time": a.time, "resume": a.resume or None})


def evaluate_main(model_dir):
    cfg = load_config(model_dir)
    p = argparse.ArgumentParser(description=f"Evaluate {cfg['model_name']} on the original val/test split.")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--weights", type=str, help="Defaults to runs/<run_name>/weights/best.pt")
    p.add_argument("--device", type=str, default="cpu", help="Device; latency is reported for this device")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for tree counting")
    p.add_argument("--skip-latency", action="store_true")
    a = p.parse_args()
    evaluate(cfg, a.split, a.weights, a.device, a.batch, a.conf, a.skip_latency)


def predict_main(model_dir):
    cfg = load_config(model_dir)
    p = argparse.ArgumentParser(description=f"Count trees with {cfg['model_name']} on an image or folder.")
    p.add_argument("source", help="Image file or folder")
    p.add_argument("--weights", type=str, help="Defaults to runs/<run_name>/weights/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no-save", action="store_true", help="Do not save annotated images")
    a = p.parse_args()
    predict(cfg, a.source, a.weights, a.conf, a.device, save=not a.no_save)
