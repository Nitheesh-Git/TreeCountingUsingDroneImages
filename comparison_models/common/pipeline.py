"""
Shared training / evaluation / counting logic for the comparison models.

Each model folder (yolov9/, yolov12/) holds a config.yaml plus thin train.py,
evaluate.py and predict.py wrappers that call into this module, so both models
are trained and measured with exactly the same code path.

Data view
---------
Training reads the half-size subset through a generated dataset "view" in
comparison_models/data/tree_top_view_half/ (git-ignored). The view contains
per-file symlinks (copies if symlinks are unavailable) to the original images
and labels, so:
  * the original dataset and its data.yaml are never modified,
  * Ultralytics writes its *.cache files inside the view, not next to the
    original labels (which would clobber the baselines' full-train cache),
  * valid/test are the original, unchanged splits.
"""

import csv
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = Path(__file__).resolve().parent
DATASET_DIR = REPO_ROOT / "datasets" / "Tree-Top-View"
MANIFEST = COMMON_DIR / "subset_manifest.txt"
VIEW_DIR = REPO_ROOT / "comparison_models" / "data" / "tree_top_view_half"


# --------------------------------------------------------------------------- #
# Config and paths
# --------------------------------------------------------------------------- #
def load_config(model_dir):
    model_dir = Path(model_dir).resolve()
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    cfg["model_dir"] = model_dir
    cfg["runs_dir"] = model_dir / "runs"
    cfg["weights_dir"] = model_dir / "weights"
    cfg["metrics_dir"] = model_dir / "metrics"
    cfg["predictions_dir"] = model_dir / "predictions"
    return cfg


def run_dir(cfg):
    return cfg["runs_dir"] / cfg["run_name"]


def best_weights(cfg):
    return run_dir(cfg) / "weights" / "best.pt"


def pretrained_weights(cfg):
    """Return the pretrained checkpoint path, downloading the official
    Ultralytics asset into the model's git-ignored weights/ folder if needed."""
    path = cfg["weights_dir"] / cfg["pretrained"]
    if not path.exists():
        from ultralytics.utils.downloads import attempt_download_asset

        cfg["weights_dir"].mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Downloading pretrained weights {cfg['pretrained']} -> {path}")
        attempt_download_asset(str(path))
    if not path.exists():
        sys.exit(f"[ERROR] Pretrained weights not available at {path}")
    return path


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def _link(src, dst):
    try:
        os.symlink(src, dst)
        return "symlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def build_data_view(force=False):
    """Create (or reuse) the dataset view and return the path to its data.yaml."""
    data_yaml = VIEW_DIR / "data.yaml"
    stamp = VIEW_DIR / ".manifest_md5"
    if not force and data_yaml.exists() and stamp.exists() and stamp.read_text() == _md5(MANIFEST):
        return data_yaml

    # Build in a private temp dir and rename into place, so two training
    # processes started together cannot trip over each other.
    final_dir = VIEW_DIR
    build_dir = VIEW_DIR.with_name(f"{VIEW_DIR.name}.tmp{os.getpid()}")
    return _build_view_into(build_dir, final_dir, force)


def _build_view_into(build_dir, final_dir, force):
    VIEW_DIR = build_dir  # noqa: N806 - files below are written into the temp dir
    data_yaml = VIEW_DIR / "data.yaml"
    stamp = VIEW_DIR / ".manifest_md5"
    if VIEW_DIR.exists():
        shutil.rmtree(VIEW_DIR)

    original = yaml.safe_load((DATASET_DIR / "data.yaml").read_text())
    train_images = [REPO_ROOT / line for line in MANIFEST.read_text().split()]
    splits = {
        "train": train_images,
        "valid": sorted((DATASET_DIR / "valid" / "images").glob("*.jpg")),
        "test": sorted((DATASET_DIR / "test" / "images").glob("*.jpg")),
    }
    methods = set()
    for split, images in splits.items():
        (VIEW_DIR / split / "images").mkdir(parents=True)
        (VIEW_DIR / split / "labels").mkdir(parents=True)
        for img in images:
            lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
            if not lbl.exists():
                sys.exit(f"[ERROR] Missing label for {img}")
            methods.add(_link(img.resolve(), VIEW_DIR / split / "images" / img.name))
            methods.add(_link(lbl.resolve(), VIEW_DIR / split / "labels" / lbl.name))

    view_yaml = {
        "path": str(final_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": original["nc"],
        "names": original["names"],
    }
    data_yaml.write_text(yaml.safe_dump(view_yaml, sort_keys=False))
    stamp.write_text(_md5(MANIFEST))

    if final_dir.exists() and force:
        shutil.rmtree(final_dir)  # only ever removes our generated view
    if final_dir.exists():
        # Another process finished first (or a stale view with an old manifest).
        if (final_dir / ".manifest_md5").exists() and \
                (final_dir / ".manifest_md5").read_text() == _md5(MANIFEST):
            shutil.rmtree(build_dir)
            return final_dir / "data.yaml"
        shutil.rmtree(final_dir)
    os.replace(build_dir, final_dir)
    print(f"[INFO] Built dataset view ({', '.join(sorted(methods))}): "
          f"train={len(splits['train'])} valid={len(splits['valid'])} test={len(splits['test'])}")
    return final_dir / "data.yaml"


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment_info():
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cpu": cpu_name(),
        "cuda_available": torch.cuda.is_available(),
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n")
    print(f"[INFO] Wrote {path.relative_to(REPO_ROOT)}")


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def _best_epoch_from_results(results_csv):
    """Best epoch using Ultralytics' own fitness (8.4.x: mAP50-95 only, see
    ultralytics.utils.metrics.DetMetrics.fitness), i.e. the epoch saved as best.pt."""
    with open(results_csv) as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not rows:
        return None, 0

    def fitness(r):
        return float(r["metrics/mAP50-95(B)"])

    best = max(rows, key=fitness)
    return {
        "epoch": int(float(best["epoch"])),
        "precision": float(best["metrics/precision(B)"]),
        "recall": float(best["metrics/recall(B)"]),
        "mAP50": float(best["metrics/mAP50(B)"]),
        "mAP50-95": float(best["metrics/mAP50-95(B)"]),
    }, len(rows)


def train(cfg, overrides):
    from ultralytics import YOLO

    t = {**cfg["train"], **{k: v for k, v in overrides.items() if v is not None}}
    data_yaml = build_data_view()
    weights = pretrained_weights(cfg)
    out = run_dir(cfg)
    resume = t.pop("resume", False)

    print("=" * 70)
    print(f"{cfg['model_name']} - comparison training on the half-size subset")
    print("=" * 70)
    print(f"Pretrained weights : {weights.relative_to(REPO_ROOT)}")
    print(f"Dataset view       : {data_yaml.relative_to(REPO_ROOT)}")
    print(f"Train images       : {len(MANIFEST.read_text().split())} (subset manifest)")
    for k, v in t.items():
        print(f"{k:<19}: {v}")
    print(f"Output directory   : {out.relative_to(REPO_ROOT)}")
    print("=" * 70)

    start = time.time()
    if resume:
        last = out / "weights" / "last.pt"
        if not last.exists():
            sys.exit(f"[ERROR] Cannot resume, {last} not found")
        YOLO(str(last)).train(resume=True)
    else:
        YOLO(str(weights)).train(
            data=str(data_yaml),
            project=str(cfg["runs_dir"]),
            name=cfg["run_name"],
            exist_ok=True,
            pretrained=True,
            verbose=True,
            val=True,
            plots=True,
            **t,
        )
    duration = time.time() - start

    best, epochs_done = _best_epoch_from_results(out / "results.csv")
    summary = {
        "model": cfg["model_name"],
        "pretrained": cfg["pretrained"],
        "train_images": len(MANIFEST.read_text().split()),
        "settings": t,
        "epochs_completed": epochs_done,
        "stop_reason": (f"time budget ({t['time']} h)" if t.get("time")
                        else "early stopping" if epochs_done < t["epochs"] else "max epochs"),
        "best_epoch_val": best,
        "training_duration_s": round(duration, 1),
        "training_duration_h": round(duration / 3600, 3),
        "resumed": resume,
        "environment": environment_info(),
        "run_dir": str(out.relative_to(REPO_ROOT)),
    }
    write_json(cfg["metrics_dir"] / "train_summary.json", summary)
    print(f"[DONE] {epochs_done} epochs in {duration / 60:.1f} min; best epoch {best and best['epoch']}")
    return summary


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def detection_metrics(cfg, weights, split, device, batch):
    from ultralytics import YOLO

    data_yaml = build_data_view()
    m = YOLO(str(weights)).val(
        data=str(data_yaml),
        split=split,
        imgsz=cfg["train"]["imgsz"],
        batch=batch,
        device=device,
        project=str(cfg["runs_dir"]),
        name=f"eval_{split}",
        exist_ok=True,
        plots=True,
        verbose=False,
    )
    return {
        "precision": round(float(m.box.mp), 4),
        "recall": round(float(m.box.mr), 4),
        "mAP50": round(float(m.box.map50), 4),
        "mAP50-95": round(float(m.box.map), 4),
        "ultralytics_speed_ms": {k: round(v, 2) for k, v in m.speed.items()},
    }


def gt_count(img_path):
    lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    return sum(1 for line in lbl.read_text().splitlines() if line.strip()) if lbl.exists() else 0


def counting_metrics(cfg, model, split, conf, device, csv_path):
    """Tree count = number of boxes at `conf`, same rule as src/evaluate_counting.py."""
    images = sorted((DATASET_DIR / ("valid" if split == "val" else split) / "images").glob("*.jpg"))
    rows, errors = [], []
    for img in images:
        gt = gt_count(img)
        pred = len(model.predict(str(img), conf=conf, imgsz=cfg["train"]["imgsz"],
                                 device=device, verbose=False)[0].boxes)
        errors.append(pred - gt)
        rows.append((img.name, gt, pred, pred - gt, abs(pred - gt)))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "ground_truth", "predicted", "error", "abs_error"])
        w.writerows(rows)

    total_gt = sum(r[1] for r in rows)
    total_pred = sum(r[2] for r in rows)
    return {
        "images": len(rows),
        "conf_threshold": conf,
        "total_ground_truth": total_gt,
        "total_predicted": total_pred,
        "net_difference": total_pred - total_gt,
        "MAE": round(sum(abs(e) for e in errors) / len(errors), 4),
        "RMSE": round((sum(e * e for e in errors) / len(errors)) ** 0.5, 4),
        "mean_error_bias": round(sum(errors) / len(errors), 4),
        "per_image_csv": str(csv_path.relative_to(REPO_ROOT)),
    }


def latency_benchmark(cfg, model, device, n_images=30, repeats=3, warmup=5):
    """Same protocol as docs/yolov8n_vs_yolo11n_comparison.md: 30 test images x 3
    passes after warm-up, batch 1, at 416 px. Only comparable across runs on the
    same hardware."""
    import torch

    images = sorted((DATASET_DIR / "test" / "images").glob("*.jpg"))[:n_images]
    kw = dict(imgsz=cfg["train"]["imgsz"], device=device, verbose=False)
    for img in images[:warmup]:
        model.predict(str(img), **kw)
    inference, total = [], []
    for _ in range(repeats):
        for img in images:
            r = model.predict(str(img), **kw)[0]
            inference.append(r.speed["inference"])
            total.append(sum(r.speed.values()))
    return {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "passes": len(inference),
        "inference_ms_mean": round(statistics.mean(inference), 2),
        "inference_ms_std": round(statistics.stdev(inference), 2),
        "total_ms_mean": round(statistics.mean(total), 2),
        "total_ms_std": round(statistics.stdev(total), 2),
        "hardware": cpu_name(),
    }


def evaluate(cfg, split, weights=None, device="cpu", batch=8, conf=0.25, skip_latency=False):
    from ultralytics import YOLO

    weights = Path(weights) if weights else best_weights(cfg)
    if not weights.exists():
        sys.exit(f"[ERROR] Weights not found: {weights}")

    print("=" * 70)
    print(f"{cfg['model_name']} - evaluation on the original '{split}' split")
    print("=" * 70)

    result = {
        "model": cfg["model_name"],
        "weights": str(weights.relative_to(REPO_ROOT)) if weights.is_relative_to(REPO_ROOT) else str(weights),
        "split": split,
        "detection": detection_metrics(cfg, weights, split, device, batch),
    }
    model = YOLO(str(weights))
    result["counting"] = counting_metrics(
        cfg, model, split, conf, device, cfg["metrics_dir"] / f"{split}_counting.csv")
    if not skip_latency:
        result["latency"] = latency_benchmark(cfg, model, device)
    result["environment"] = environment_info()

    write_json(cfg["metrics_dir"] / f"{split}_metrics.json", result)
    d, c = result["detection"], result["counting"]
    print(f"Precision {d['precision']:.4f} | Recall {d['recall']:.4f} | "
          f"mAP50 {d['mAP50']:.4f} | mAP50-95 {d['mAP50-95']:.4f} | count MAE {c['MAE']:.4f}")
    if "latency" in result:
        l = result["latency"]
        print(f"Latency ({l['device']}): inference {l['inference_ms_mean']} ± {l['inference_ms_std']} ms, "
              f"total {l['total_ms_mean']} ± {l['total_ms_std']} ms")
    return result


# --------------------------------------------------------------------------- #
# Prediction / counting
# --------------------------------------------------------------------------- #
def predict(cfg, source, weights=None, conf=0.25, device="cpu", save=True):
    from ultralytics import YOLO

    weights = Path(weights) if weights else best_weights(cfg)
    if not weights.exists():
        sys.exit(f"[ERROR] Weights not found: {weights}")
    results = YOLO(str(weights)).predict(
        source=str(source),
        conf=conf,
        imgsz=cfg["train"]["imgsz"],
        device=device,
        save=save,
        project=str(cfg["predictions_dir"]),
        name="predict",
        exist_ok=True,
        verbose=False,
    )
    for r in results:
        print(f"{Path(r.path).name}: {len(r.boxes)} trees (conf >= {conf})")
    if save and results:
        print(f"[INFO] Annotated images saved to {Path(results[0].save_dir).relative_to(REPO_ROOT)}")
    return results
