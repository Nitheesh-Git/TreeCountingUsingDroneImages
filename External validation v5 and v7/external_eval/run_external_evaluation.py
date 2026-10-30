import os
import sys
import csv
import json
import math
import time
import shutil
import subprocess
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEST100_DIR = PROJECT_ROOT / "External validation v5 and v7" / "testdataset100"
EXT_EVAL_DIR = PROJECT_ROOT / "External validation v5 and v7" / "external_eval"
DATASET_LINK_DIR = EXT_EVAL_DIR / "selvabox_yolo"
IMAGES_LINK = DATASET_LINK_DIR / "images"
LABELS_DIR = DATASET_LINK_DIR / "labels"

def setup_eval_staging():
    # Clean up any accidental cache in testdataset100
    for accidental_cache in TEST100_DIR.glob("*.cache"):
        accidental_cache.unlink(missing_ok=True)

    # Remove old dataset_link junction if present
    old_junc = EXT_EVAL_DIR / "dataset_link" / "images"
    if old_junc.exists():
        subprocess.run(["cmd", "/c", "rmdir", str(old_junc)], capture_output=True)
        shutil.rmtree(EXT_EVAL_DIR / "dataset_link", ignore_errors=True)

    EXT_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_LINK_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_LINK.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    # Create NTFS hardlinks for the 100 PNGs so Path.resolve() stays inside DATASET_LINK_DIR
    for src_png in sorted((TEST100_DIR / "images").glob("image_*.png")):
        dst_png = IMAGES_LINK / src_png.name
        if not dst_png.exists():
            os.link(src_png, dst_png)

    # Copy 100 .txt label files into LABELS_DIR
    src_ann = TEST100_DIR / "annotations"
    txt_files = sorted(src_ann.glob("image_*.txt"))
    for tf in txt_files:
        dst = LABELS_DIR / tf.name
        if not dst.exists() or dst.stat().st_size != tf.stat().st_size:
            shutil.copy2(tf, dst)

    # Remove stale cache in DATASET_LINK_DIR if any
    for cache_f in DATASET_LINK_DIR.glob("*.cache"):
        cache_f.unlink(missing_ok=True)

    # Create Ultralytics YAML
    ultra_yaml = EXT_EVAL_DIR / "selvabox_test100.yaml"
    ultra_yaml.write_text(
        f"path: {DATASET_LINK_DIR.as_posix()}\n"
        f"train: images\n"
        f"val: images\n"
        f"test: images\n\n"
        f"nc: 1\n"
        f"names: ['tree-top']\n",
        encoding="utf-8",
    )

    # Create YOLOv7 YAML
    v7_yaml = EXT_EVAL_DIR / "selvabox_test100_yolov7.yaml"
    v7_yaml.write_text(
        f"train: {(DATASET_LINK_DIR / 'images').as_posix()}\n"
        f"val: {(DATASET_LINK_DIR / 'images').as_posix()}\n"
        f"test: {(DATASET_LINK_DIR / 'images').as_posix()}\n\n"
        f"nc: 1\n"
        f"names: ['tree-top']\n",
        encoding="utf-8",
    )
    print(f"[INFO] Staging ready: {len(list(IMAGES_LINK.glob('*.png')))} images, {len(list(LABELS_DIR.glob('*.txt')))} labels.")
    return ultra_yaml, v7_yaml


def load_manifest_and_gt():
    manifest_path = TEST100_DIR / "manifest.csv"
    records = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["local_filename"]
            stem = Path(fname).stem
            label_path = LABELS_DIR / f"{stem}.txt"
            boxes = []
            W, H = int(row["image_width"]), int(row["image_height"])
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, xc, yc, w, h = map(float, parts)
                            x1 = (xc - w / 2.0) * W
                            y1 = (yc - h / 2.0) * H
                            x2 = (xc + w / 2.0) * W
                            y2 = (yc + h / 2.0) * H
                            boxes.append([x1, y1, x2, y2])
            records[fname] = {
                "new_filename": fname,
                "original_filename": row["original_filename"],
                "source_site": row["source_site"],
                "source_raster": row["source_image_id"].split("/")[1] if "/" in row["source_image_id"] else row["source_site"],
                "width": W,
                "height": H,
                "gt_count": len(boxes),
                "gt_boxes": np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            }
    return records


def greedy_iou_match(pred_boxes, pred_scores, gt_boxes, iou_Half=0.50):
    """Greedy matching sorted by descending confidence score at IoU >= 0.50."""
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)
    if n_pred == 0:
        return 0, 0, n_gt
    if n_gt == 0:
        return 0, n_pred, 0

    order = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[order]

    # Compute pairwise IoU matrix (n_pred, n_gt)
    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_p = np.maximum(0.0, (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]))
    area_g = np.maximum(0.0, (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]))
    union = area_p[:, None] + area_g[None, :] - inter
    iou_mat = np.where(union > 0, inter / union, 0.0)

    gt_matched = np.zeros(n_gt, dtype=bool)
    tp = 0
    fp = 0
    for i in range(n_pred):
        ious = iou_mat[i]
        # mask already matched gt
        ious_unmatched = np.where(~gt_matched, ious, -1.0)
        best_gt = int(np.argmax(ious_unmatched))
        if ious_unmatched[best_gt] >= iou_Half:
            tp += 1
            gt_matched[best_gt] = True
        else:
            fp += 1
    fn = int(np.sum(~gt_matched))
    return tp, fp, fn


def compute_detailed_metrics(per_image_preds, gt_records, conf_thres):
    """Computes overall and site-stratified counting + IoU>=0.50 metrics at conf_thres."""
    sites = ["all", "ecuador_tiputini", "brazil_zf2", "panama_aguasalud"]
    accum = {
        s: {
            "num_images": 0,
            "gt_trees": 0,
            "pred_trees": 0,
            "abs_errors": [],
            "sq_errors": [],
            "signed_errors": [],
            "rel_abs_errors": [],
            "gt_list": [],
            "pred_list": [],
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "box_widths_norm": [],
            "box_heights_norm": [],
        }
        for s in sites
    }

    per_image_rows = []
    for fname in sorted(gt_records.keys()):
        rec = gt_records[fname]
        site = rec["source_site"]
        gt_count = rec["gt_count"]
        gt_boxes = rec["gt_boxes"]

        raw = per_image_preds[fname]
        mask = raw["scores"] >= conf_thres
        p_boxes = raw["boxes"][mask]
        p_scores = raw["scores"][mask]
        pred_count = int(len(p_boxes))

        tp, fp, fn = greedy_iou_match(p_boxes, p_scores, gt_boxes, iou_Half=0.50)
        err = pred_count - gt_count
        abs_err = abs(err)
        sq_err = err * err
        rel_err = abs_err / max(1, gt_count)

        if conf_thres == 0.25:
            per_image_rows.append({
                "new_filename": fname,
                "source_site": site,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "signed_error": err,
                "abs_error": abs_err,
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

        for target_s in ["all", site]:
            d = accum[target_s]
            d["num_images"] += 1
            d["gt_trees"] += gt_count
            d["pred_trees"] += pred_count
            d["abs_errors"].append(abs_err)
            d["sq_errors"].append(sq_err)
            d["signed_errors"].append(err)
            d["rel_abs_errors"].append(rel_err)
            d["gt_list"].append(gt_count)
            d["pred_list"].append(pred_count)
            d["tp"] += tp
            d["fp"] += fp
            d["fn"] += fn

    summary = {}
    for s in sites:
        d = accum[s]
        n = max(1, d["num_images"])
        mae = float(np.mean(d["abs_errors"]))
        rmse = float(math.sqrt(np.mean(d["sq_errors"])))
        bias = float(np.mean(d["signed_errors"]))
        mape = float(np.mean(d["rel_abs_errors"]) * 100.0)
        median_ae = float(np.median(d["abs_errors"]))
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        if len(d["gt_list"]) > 1 and np.std(d["pred_list"]) > 0 and np.std(d["gt_list"]) > 0:
            corr = float(np.corrcoef(d["gt_list"], d["pred_list"])[0, 1])
        else:
            corr = 0.0
        net_diff = d["pred_trees"] - d["gt_trees"]
        net_pct = (net_diff / d["gt_trees"] * 100.0) if d["gt_trees"] > 0 else 0.0
        summary[s] = {
            "num_images": d["num_images"],
            "gt_trees": d["gt_trees"],
            "pred_trees": d["pred_trees"],
            "net_count_diff": net_diff,
            "net_count_bias_pct": round(net_pct, 2),
            "counting_mae": round(mae, 4),
            "counting_rmse": round(rmse, 4),
            "counting_median_ae": round(median_ae, 4),
            "counting_bias_per_img": round(bias, 4),
            "counting_mape_pct": round(mape, 2),
            "pearson_r": round(corr, 4),
            "tp_iou50": tp,
            "fp_iou50": fp,
            "fn_iou50": fn,
            "precision_iou50": round(prec, 4),
            "recall_iou50": round(rec, 4),
            "f1_iou50": round(f1, 4),
        }
    return summary, per_image_rows


def evaluate_ultralytics_model(model_key, weights_path, ultra_yaml, gt_records, imgsz=416):
    from ultralytics import YOLO
    print(f"\n{'='*70}\nEvaluating {model_key} at imgsz={imgsz} ({weights_path})\n{'='*70}")
    model = YOLO(str(weights_path))
    run_name = f"{model_key.lower()}_ext_val_{imgsz}"
    metrics = model.val(
        data=str(ultra_yaml),
        split="test",
        imgsz=imgsz,
        batch=8,
        device="cpu",
        workers=0,
        project=str(EXT_EVAL_DIR / "runs"),
        name=run_name,
        exist_ok=True,
        plots=(imgsz == 416),
        verbose=False,
    )
    sp = metrics.speed
    val_summary = {
        "model": model_key,
        "imgsz": imgsz,
        "batch": 8,
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "preprocess_ms": round(float(sp.get("preprocess", 0.0)), 2),
        "inference_ms": round(float(sp.get("inference", 0.0)), 2),
        "postprocess_ms": round(float(sp.get("postprocess", 0.0)), 2),
        "total_ms": round(float(sp.get("preprocess", 0.0) + sp.get("inference", 0.0) + sp.get("postprocess", 0.0)), 2),
    }
    print(f"[VAL {model_key} @{imgsz}] P={val_summary['precision']:.4f}, R={val_summary['recall']:.4f}, mAP50={val_summary['mAP50']:.4f}, mAP50-95={val_summary['mAP50_95']:.4f}, total={val_summary['total_ms']:.2f}ms")

    # Per-image predictions at conf=0.10 so we can threshold at 0.10, 0.15, 0.25
    per_image_preds = {}
    img_paths = [str(IMAGES_LINK / fname) for fname in sorted(gt_records.keys())]
    t0 = time.perf_counter()
    results = model.predict(source=img_paths, imgsz=imgsz, conf=0.10, iou=0.45, device="cpu", verbose=False, stream=True)
    for res in results:
        fname = Path(res.path).name
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            scores = np.zeros((0,), dtype=np.float32)
        per_image_preds[fname] = {"boxes": boxes, "scores": scores}
    elapsed_pred = (time.perf_counter() - t0) * 1000.0 / len(img_paths)
    val_summary["single_stream_ms_per_img"] = round(elapsed_pred, 2)

    conf_summaries = {}
    per_image_25 = []
    for c in [0.25, 0.15, 0.10]:
        s_dict, rows_25 = compute_detailed_metrics(per_image_preds, gt_records, c)
        conf_summaries[f"conf_{c:.2f}"] = s_dict
        if c == 0.25:
            per_image_25 = rows_25

    val_summary["counting_and_site_breakdown"] = conf_summaries
    return val_summary, per_image_25


def evaluate_yolov7_tiny_subproc(v7_yaml, imgsz=416):
    """Runs YOLOv7-tiny evaluation in an isolated subprocess so yolov7_repo's 'models'/'utils' don't collide with ultralytics."""
    helper_script = EXT_EVAL_DIR / "_run_v7_helper.py"
    out_json = EXT_EVAL_DIR / f"_v7_result_{imgsz}.json"
    helper_code = f'''import os, sys, json, time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(r"{str(PROJECT_ROOT)}")
YOLOV7_DIR = PROJECT_ROOT / "External validation v5 and v7" / "yolov7_tiny"
SHIMS_DIR = YOLOV7_DIR / "shims"
YOLOV7_REPO = YOLOV7_DIR / "yolov7_repo"
sys.path.insert(0, str(SHIMS_DIR))
sys.path.insert(0, str(YOLOV7_REPO))

import torch
import cv2
import yaml
from types import SimpleNamespace
import test as yolov7_test
from models.experimental import attempt_load
from utils.datasets import create_dataloader, letterbox
from utils.general import check_dataset, check_img_size, non_max_suppression, scale_coords

weights = str(YOLOV7_DIR / "runs" / "yolov7_tiny_subset" / "weights" / "best.pt")
data_yaml = r"{str(v7_yaml)}"
imgsz = {imgsz}
save_dir = Path(r"{str(EXT_EVAL_DIR / 'runs' / f'yolov7_tiny_ext_val_{imgsz}')}")
save_dir.mkdir(parents=True, exist_ok=True)

opt = SimpleNamespace(
    data=data_yaml,
    weights=[weights],
    batch_size=8,
    img_size=imgsz,
    conf_thres=0.001,
    iou_thres=0.65,
    task="test",
    device="cpu",
    single_cls=True,
    augment=False,
    verbose=False,
    save_txt=False,
    save_hybrid=False,
    save_conf=False,
    save_json=False,
    project=str(save_dir.parent),
    name=save_dir.name,
    exist_ok=True,
    no_trace=True,
    v5_metric=True,
)

model = attempt_load(weights, map_location="cpu")
gs = max(int(model.stride.max()), 32)
imgsz = check_img_size(imgsz, s=gs)
with open(data_yaml, "r", encoding="utf-8") as f:
    data_dict = yaml.safe_load(f)
check_dataset(data_dict)

dataloader = create_dataloader(
    data_dict["test"],
    imgsz,
    8,
    gs,
    opt,
    pad=0.5,
    rect=True,
    workers=0,
    prefix="test: ",
)[0]

results, maps, times = yolov7_test.test(
    data_dict,
    batch_size=8,
    imgsz=imgsz,
    conf_thres=0.001,
    iou_thres=0.65,
    save_json=False,
    single_cls=True,
    augment=False,
    verbose=False,
    model=model,
    dataloader=dataloader,
    save_dir=save_dir,
    plots=({imgsz == 416}),
    compute_loss=None,
    half_precision=False,
    trace=False,
    v5_metric=True,
)
mp, mr, map50, map50_95 = results[0], results[1], results[2], results[3]
inf_ms, nms_ms, tot_ms = times[0], times[1], times[2]

# Per-image predictions at conf>=0.10, iou=0.45
img_dir = Path(r"{str(IMAGES_LINK)}")
img_files = sorted(img_dir.glob("image_*.png"))
per_image_preds = {{}}
t0 = time.perf_counter()
for img_path in img_files:
    img0 = cv2.imread(str(img_path))
    h0, w0 = img0.shape[:2]
    img = letterbox(img0, imgsz, stride=gs)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to("cpu").float() / 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)
    with torch.no_grad():
        pred = model(im, augment=False)[0]
        pred = non_max_suppression(pred, 0.10, 0.45, classes=None, agnostic=False)
    det = pred[0]
    if det is not None and len(det):
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
        boxes = det[:, :4].cpu().numpy().tolist()
        scores = det[:, 4].cpu().numpy().tolist()
    else:
        boxes = []
        scores = []
    per_image_preds[img_path.name] = {{"boxes": boxes, "scores": scores}}
stream_ms = (time.perf_counter() - t0) * 1000.0 / len(img_files)

out = {{
    "val_summary": {{
        "model": "YOLOv7-tiny",
        "imgsz": imgsz,
        "batch": 8,
        "precision": round(float(mp), 4),
        "recall": round(float(mr), 4),
        "mAP50": round(float(map50), 4),
        "mAP50_95": round(float(map50_95), 4),
        "preprocess_ms": 0.0,
        "inference_ms": round(float(inf_ms), 2),
        "postprocess_ms": round(float(nms_ms), 2),
        "total_ms": round(float(tot_ms), 2),
        "single_stream_ms_per_img": round(float(stream_ms), 2),
    }},
    "per_image_preds": per_image_preds,
}}
with open(r"{str(out_json)}", "w", encoding="utf-8") as f:
    json.dump(out, f)
'''
    helper_script.write_text(helper_code, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    subprocess.run([sys.executable, str(helper_script)], check=True, env=env)
    data = json.loads(out_json.read_text(encoding="utf-8"))
    helper_script.unlink(missing_ok=True)
    out_json.unlink(missing_ok=True)
    return data


def main():
    ultra_yaml, v7_yaml = setup_eval_staging()
    gt_records = load_manifest_and_gt()

    models_ultra = [
        ("YOLOv8n", PROJECT_ROOT / "runs" / "detect" / "tree_detection" / "baseline_run" / "weights" / "best.pt"),
        ("YOLO11n", PROJECT_ROOT / "runs" / "detect" / "tree_detection" / "yolov11n_phase2" / "weights" / "best.pt"),
        ("YOLOv5n", PROJECT_ROOT / "External validation v5 and v7" / "yolov5" / "runs" / "yolov5n_subset" / "weights" / "best.pt"),
    ]

    all_results = {"imgsz_416": {}, "imgsz_640": {}}
    per_image_merged_416 = {}

    for fname, rec in sorted(gt_records.items()):
        per_image_merged_416[fname] = {
            "new_filename": fname,
            "original_filename": rec["original_filename"],
            "source_site": rec["source_site"],
            "source_raster": rec["source_raster"],
            "gt_count": rec["gt_count"],
        }

    for imgsz in [416, 640]:
        key_sz = f"imgsz_{imgsz}"
        for m_name, w_path in models_ultra:
            val_sum, rows_25 = evaluate_ultralytics_model(m_name, w_path, ultra_yaml, gt_records, imgsz=imgsz)
            all_results[key_sz][m_name] = val_sum
            if imgsz == 416:
                for r in rows_25:
                    fn = r["new_filename"]
                    per_image_merged_416[fn][f"{m_name}_pred"] = r["pred_count"]
                    per_image_merged_416[fn][f"{m_name}_err"] = r["signed_error"]
                    per_image_merged_416[fn][f"{m_name}_abs_err"] = r["abs_error"]
                    per_image_merged_416[fn][f"{m_name}_tp"] = r["tp"]
                    per_image_merged_416[fn][f"{m_name}_fp"] = r["fp"]
                    per_image_merged_416[fn][f"{m_name}_fn"] = r["fn"]

        # YOLOv7-tiny
        print(f"\n{'='*70}\nEvaluating YOLOv7-tiny at imgsz={imgsz}\n{'='*70}")
        v7_raw = evaluate_yolov7_tiny_subproc(v7_yaml, imgsz=imgsz)
        v7_val = v7_raw["val_summary"]
        v7_preds = {
            k: {
                "boxes": np.array(v["boxes"], dtype=np.float32) if v["boxes"] else np.zeros((0, 4), dtype=np.float32),
                "scores": np.array(v["scores"], dtype=np.float32) if v["scores"] else np.zeros((0,), dtype=np.float32),
            }
            for k, v in v7_raw["per_image_preds"].items()
        }
        conf_summaries = {}
        rows_25_v7 = []
        for c in [0.25, 0.15, 0.10]:
            s_dict, r25 = compute_detailed_metrics(v7_preds, gt_records, c)
            conf_summaries[f"conf_{c:.2f}"] = s_dict
            if c == 0.25:
                rows_25_v7 = r25
        v7_val["counting_and_site_breakdown"] = conf_summaries
        all_results[key_sz]["YOLOv7-tiny"] = v7_val
        print(f"[VAL YOLOv7-tiny @{imgsz}] P={v7_val['precision']:.4f}, R={v7_val['recall']:.4f}, mAP50={v7_val['mAP50']:.4f}, mAP50-95={v7_val['mAP50_95']:.4f}, total={v7_val['total_ms']:.2f}ms")

        if imgsz == 416:
            for r in rows_25_v7:
                fn = r["new_filename"]
                per_image_merged_416[fn]["YOLOv7-tiny_pred"] = r["pred_count"]
                per_image_merged_416[fn]["YOLOv7-tiny_err"] = r["signed_error"]
                per_image_merged_416[fn]["YOLOv7-tiny_abs_err"] = r["abs_error"]
                per_image_merged_416[fn]["YOLOv7-tiny_tp"] = r["tp"]
                per_image_merged_416[fn]["YOLOv7-tiny_fp"] = r["fp"]
                per_image_merged_416[fn]["YOLOv7-tiny_fn"] = r["fn"]

    # Save master JSON
    summary_json_path = EXT_EVAL_DIR / "external_evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save per-image CSV for imgsz=416, conf=0.25
    csv_path = EXT_EVAL_DIR / "per_image_external_counts_416.csv"
    fieldnames = list(next(iter(per_image_merged_416.values())).keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fn in sorted(per_image_merged_416.keys()):
            writer.writerow(per_image_merged_416[fn])

    print(f"\n[SUCCESS] Saved master external evaluation JSON to: {summary_json_path}")
    print(f"[SUCCESS] Saved 100-image per-image CSV to: {csv_path}")


if __name__ == "__main__":
    main()
