"""
Zero-shot evaluation of the trained comparison models on the external SelvaBox
`testdataset100` (no training, no tuning).

Pre-specified settings (fixed before looking at any result):
  * imgsz 832 (primary): matches the training ground resolution. Tree-Top-View
    tiles are NEON 400 px at 10 cm, stretched to 416 px (about 9.6 cm/px); a
    SelvaBox tile covers 80 m, so 80 / 0.096 = 833, rounded to the stride-32 832.
  * imgsz 416 (secondary): the plain training input size.
  * conf 0.25 for counting / TP-FP-FN, the same as on the internal test split.
  * max_det 1000: SelvaBox tiles hold up to 361 crowns, above the default 300.

The dataset keeps labels in annotations/, so a symlink view with an
images/ + labels/ layout is built in External Validation v9 and v12/data/ (git-ignored);
the extracted dataset itself is only read.

Usage:
    python "External Validation v9 and v12/common/evaluate_external.py" [--imgsz 832 416]
"""

import argparse
import csv
import os
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import PROJECT_DIR, REPO_ROOT, best_weights, environment_info, load_config, write_json  # noqa: E402
from report_details import gt_boxes, iou  # noqa: E402

DATASET = PROJECT_DIR / "external_dataset" / "testdataset100"
VIEW = PROJECT_DIR / "data" / "testdataset100_view"
MODEL_DIRS = [PROJECT_DIR / "yolov9", PROJECT_DIR / "yolov12"]
CONF = 0.25
MAX_DET = 1000


def build_view():
    images = sorted((DATASET / "images").glob("*.png"))
    if len(images) != 100:
        sys.exit(f"[ERROR] expected 100 images in {DATASET / 'images'}, found {len(images)}")
    tmp = VIEW.with_name(f"{VIEW.name}.tmp{os.getpid()}")
    for sub in ("images", "labels"):
        (tmp / sub).mkdir(parents=True)
    for img in images:
        lbl = DATASET / "annotations" / f"{img.stem}.txt"
        if not lbl.exists():
            sys.exit(f"[ERROR] missing annotation for {img.name}")
        os.symlink(img.resolve(), tmp / "images" / img.name)
        os.symlink(lbl.resolve(), tmp / "labels" / lbl.name)
    (tmp / "data.yaml").write_text(yaml.safe_dump({
        "path": str(VIEW), "train": "images", "val": "images", "test": "images",
        "nc": 1, "names": ["tree-top"]}, sort_keys=False))
    if VIEW.exists():
        shutil.rmtree(VIEW)  # only ever removes our generated view
    os.replace(tmp, VIEW)
    return VIEW / "data.yaml"


def site_map():
    with open(DATASET / "manifest.csv") as f:
        return {r["local_filename"]: r["source_site"] for r in csv.DictReader(f)}


def predict_all(model, images, imgsz):
    """One pass at conf 0.25: per-image boxes for counting and TP/FP/FN, plus timing."""
    preds, times = {}, []
    for img in images:
        t0 = time.perf_counter()
        r = model.predict(str(img), conf=CONF, imgsz=imgsz, max_det=MAX_DET, device="cpu", verbose=False)[0]
        times.append((time.perf_counter() - t0) * 1000)
        preds[img.name] = (sorted(([list(map(int, b.xyxy[0])), float(b.conf[0])] for b in r.boxes),
                                  key=lambda p: p[1], reverse=True), r.speed)
    return preds, times


def greedy_match(preds, gts, thr=0.5):
    matched, tp = set(), 0
    for box, _ in preds:
        best, idx = 0.0, -1
        for gi, g in enumerate(gts):
            if gi not in matched:
                v = iou(box, g)
                if v > best:
                    best, idx = v, gi
        if best >= thr:
            matched.add(idx)
            tp += 1
    return tp, len(preds) - tp, len(gts) - len(matched)


def summarise(rows):
    n = len(rows)
    errs = [r["error"] for r in rows]
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("TP", "FP", "FN"))
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    gt, pred = sum(r["gt"] for r in rows), sum(r["pred"] for r in rows)
    return {
        "images": n, "total_ground_truth": gt, "total_predicted": pred, "net_difference": pred - gt,
        "net_difference_pct": round(100 * (pred - gt) / gt, 2) if gt else None,
        "MAE": round(sum(abs(e) for e in errs) / n, 3),
        "RMSE": round((sum(e * e for e in errs) / n) ** 0.5, 3),
        "bias": round(sum(errs) / n, 3),
        "mean_relative_error": round(statistics.mean(abs(r["error"]) / r["gt"] for r in rows if r["gt"]), 4),
        "TP": tp, "FP": fp, "FN": fn,
        "precision_conf025": round(p, 4), "recall_conf025": round(rc, 4),
        "F1_conf025": round(2 * p * rc / (p + rc), 4) if p + rc else 0.0,
    }


def evaluate(cfg, data_yaml, imgsz, sites):
    from ultralytics import YOLO

    weights = best_weights(cfg)
    images = sorted((VIEW / "images").glob("*.png"))
    print(f"\n=== {cfg['model_name']} @ imgsz {imgsz} ===")

    m = YOLO(str(weights)).val(data=str(data_yaml), split="test", imgsz=imgsz, batch=4, device="cpu",
                               max_det=MAX_DET, project=str(cfg["runs_dir"]), name=f"external_{imgsz}",
                               exist_ok=True, plots=True, verbose=False)
    detection = {"precision": round(float(m.box.mp), 4), "recall": round(float(m.box.mr), 4),
                 "mAP50": round(float(m.box.map50), 4), "mAP50-95": round(float(m.box.map), 4),
                 "ultralytics_speed_ms": {k: round(v, 2) for k, v in m.speed.items()}}

    model = YOLO(str(weights))
    model.predict(str(images[0]), imgsz=imgsz, device="cpu", verbose=False)  # warm-up
    preds, times = predict_all(model, images, imgsz)

    rows = []
    for img in images:
        h, w = cv2.imread(str(img)).shape[:2]
        gts = gt_boxes(img, w, h)
        boxes, _ = preds[img.name]
        tp, fp, fn = greedy_match(boxes, gts)
        rows.append({"image": img.name, "site": sites[img.name], "gt": len(gts), "pred": len(boxes),
                     "error": len(boxes) - len(gts), "TP": tp, "FP": fp, "FN": fn})

    by_site = defaultdict(list)
    for r in rows:
        by_site[r["site"]].append(r)

    csv_path = cfg["metrics_dir"] / f"external_testdataset100_imgsz{imgsz}.csv"
    with open(csv_path, "w", newline="") as f:
        wri = csv.DictWriter(f, fieldnames=list(rows[0]))
        wri.writeheader()
        wri.writerows(rows)

    inf = [s["inference"] for _, s in preds.values()]
    result = {
        "model": cfg["model_name"], "weights": str(weights.relative_to(REPO_ROOT)),
        "dataset": "SelvaBox testdataset100 (100 tiles, 1777x1777, 8,111 crowns)",
        "imgsz": imgsz, "conf_for_counting": CONF, "max_det": MAX_DET,
        "detection": detection,
        "counting_and_matching": summarise(rows),
        "by_site": {s: summarise(v) for s, v in sorted(by_site.items())},
        "latency_cpu_ms": {"inference_mean": round(statistics.mean(inf), 1),
                           "end_to_end_mean_incl_png_read": round(statistics.mean(times), 1)},
        "per_image_csv": str(csv_path.relative_to(REPO_ROOT)),
        "environment": environment_info(),
    }
    write_json(cfg["metrics_dir"] / f"external_testdataset100_imgsz{imgsz}.json", result)
    d, c = detection, result["counting_and_matching"]
    print(f"P {d['precision']:.4f} R {d['recall']:.4f} mAP50 {d['mAP50']:.4f} mAP50-95 {d['mAP50-95']:.4f} | "
          f"count {c['total_predicted']}/{c['total_ground_truth']} MAE {c['MAE']} | TP/FP/FN {c['TP']}/{c['FP']}/{c['FN']}")
    return result


def main():
    p = argparse.ArgumentParser(description="Zero-shot evaluation on SelvaBox testdataset100.")
    p.add_argument("--imgsz", type=int, nargs="+", default=[832, 416])
    a = p.parse_args()
    data_yaml = build_view()
    sites = site_map()
    for d in MODEL_DIRS:
        cfg = load_config(d)
        for s in a.imgsz:
            evaluate(cfg, data_yaml, s, sites)


if __name__ == "__main__":
    main()
