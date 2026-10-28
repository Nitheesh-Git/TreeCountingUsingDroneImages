"""
Extra report statistics for the comparison models (no training, inference only).

For each trained model (runs/<run_name>/weights/best.pt) this computes:
  * model complexity: parameters, GFLOPs at 416 px, checkpoint size
  * F1 from the test-split precision/recall
  * TP / FP / FN with the same greedy IoU>=0.5, conf=0.25 matching as
    src/analyze_predictions_yolov11.py, on the same first-30 sorted validation
    images, plus the full 55-image test split
  * extended counting statistics on the test split: exact / within +-1 / +-2
    image share, MAE by tree-density bucket and by NEON site
  * per-epoch training summary from results.csv (final losses, time/epoch)

Writes metrics/report_details.json in each model folder.

Usage:
    python comparison_models/common/report_details.py
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import DATASET_DIR, REPO_ROOT, best_weights, load_config, run_dir, write_json  # noqa: E402

MODEL_DIRS = [REPO_ROOT / "comparison_models" / "yolov9", REPO_ROOT / "comparison_models" / "yolov12"]
SITE_RE = re.compile(r"^(?:\d{4}_)?([A-Z]+)")


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / float(union) if union > 0 else 0.0


def gt_boxes(img_path, w, h):
    lbl = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    boxes = []
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                x, y, bw, bh = map(float, p[1:5])
                boxes.append([int((x - bw / 2) * w), int((y - bh / 2) * h),
                              int((x + bw / 2) * w), int((y + bh / 2) * h)])
    return boxes


def match(model, images, conf=0.25, iou_thr=0.5):
    """Identical algorithm to src/analyze_predictions_yolov11.py."""
    tp = fp = fn = 0
    per_image = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gts = gt_boxes(img_path, w, h)
        r = model(str(img_path), conf=conf, imgsz=416, verbose=False)[0]
        preds = sorted(([list(map(int, b.xyxy[0])), float(b.conf[0])] for b in r.boxes),
                       key=lambda p: p[1], reverse=True)
        matched = set()
        i_tp = 0
        for box, _ in preds:
            best, best_idx = 0.0, -1
            for gi, g in enumerate(gts):
                if gi in matched:
                    continue
                v = iou(box, g)
                if v > best:
                    best, best_idx = v, gi
            if best >= iou_thr:
                matched.add(best_idx)
                i_tp += 1
        tp += i_tp
        fp += len(preds) - i_tp
        fn += len(gts) - len(matched)
        per_image.append((img_path.name, len(gts), len(preds)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"images": len(images), "TP": tp, "FP": fp, "FN": fn,
            "precision": round(p, 4), "recall": round(r, 4),
            "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0}, per_image


def counting_breakdown(per_image):
    errs = [(name, gt, pred, pred - gt) for name, gt, pred in per_image]
    n = len(errs)

    def mae(rows):
        return round(sum(abs(e[3]) for e in rows) / len(rows), 4) if rows else None

    buckets = {"0-5 trees": [], "6-10 trees": [], "11+ trees": []}
    sites = defaultdict(list)
    for e in errs:
        buckets["0-5 trees" if e[1] <= 5 else "6-10 trees" if e[1] <= 10 else "11+ trees"].append(e)
        sites[SITE_RE.match(e[0]).group(1)].append(e)
    return {
        "exact_count_share": round(sum(1 for e in errs if e[3] == 0) / n, 4),
        "within_1_share": round(sum(1 for e in errs if abs(e[3]) <= 1) / n, 4),
        "within_2_share": round(sum(1 for e in errs if abs(e[3]) <= 2) / n, 4),
        "over_counted_images": sum(1 for e in errs if e[3] > 0),
        "under_counted_images": sum(1 for e in errs if e[3] < 0),
        "mean_relative_error": round(sum(abs(e[3]) / e[1] for e in errs if e[1]) /
                                     max(1, sum(1 for e in errs if e[1])), 4),
        "MAE_by_density": {k: {"images": len(v), "MAE": mae(v)} for k, v in buckets.items()},
        "MAE_by_site": {k: {"images": len(v), "MAE": mae(v)} for k, v in sorted(sites.items())},
    }


def training_curve(cfg):
    with open(run_dir(cfg) / "results.csv") as f:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    last = rows[-1]
    times = [float(r["time"]) for r in rows]
    return {
        "epochs": len(rows),
        "avg_epoch_time_s": round(times[-1] / len(rows), 1),
        "final_train_box_loss": float(last["train/box_loss"]),
        "final_train_cls_loss": float(last["train/cls_loss"]),
        "final_train_dfl_loss": float(last["train/dfl_loss"]),
        "final_val_box_loss": float(last["val/box_loss"]),
        "final_val_cls_loss": float(last["val/cls_loss"]),
        "final_val_dfl_loss": float(last["val/dfl_loss"]),
        "val_mAP50_by_epoch": [round(float(r["metrics/mAP50(B)"]), 4) for r in rows],
        "val_mAP50-95_by_epoch": [round(float(r["metrics/mAP50-95(B)"]), 4) for r in rows],
    }


def complexity(model, weights):
    from ultralytics.utils.torch_utils import get_flops

    params = sum(p.numel() for p in model.model.parameters())
    return {
        "parameters": params,
        "parameters_M": round(params / 1e6, 2),
        "GFLOPs_416": round(get_flops(model.model, imgsz=416), 2),
        "GFLOPs_640": round(get_flops(model.model, imgsz=640), 2),
        "layers": len(list(model.model.modules())),
        "checkpoint_MB": round(Path(weights).stat().st_size / 1e6, 2),
    }


def main():
    from ultralytics import YOLO

    val_images = sorted((DATASET_DIR / "valid" / "images").glob("*.jpg"))[:30]
    test_images = sorted((DATASET_DIR / "test" / "images").glob("*.jpg"))
    for d in MODEL_DIRS:
        cfg = load_config(d)
        w = best_weights(cfg)
        model = YOLO(str(w))
        test = json.loads((cfg["metrics_dir"] / "test_metrics.json").read_text())
        p, r = test["detection"]["precision"], test["detection"]["recall"]
        val30, _ = match(model, val_images)
        test_match, per_image = match(model, test_images)
        out = {
            "model": cfg["model_name"],
            "complexity": complexity(model, w),
            "test_F1_from_ultralytics_PR": round(2 * p * r / (p + r), 4),
            "error_analysis_val30": val30,
            "error_analysis_test55": test_match,
            "counting_test_breakdown": counting_breakdown(per_image),
            "training_curve": training_curve(cfg),
        }
        write_json(cfg["metrics_dir"] / "report_details.json", out)
        c = out["complexity"]
        print(f"{cfg['model_name']}: {c['parameters_M']}M params, {c['GFLOPs_416']} GFLOPs@416, "
              f"{c['checkpoint_MB']} MB | val30 TP/FP/FN {val30['TP']}/{val30['FP']}/{val30['FN']} | "
              f"test TP/FP/FN {test_match['TP']}/{test_match['FP']}/{test_match['FN']}")


if __name__ == "__main__":
    main()
