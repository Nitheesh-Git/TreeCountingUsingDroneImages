"""
Collect the comparison-model metrics into comparison_models/RESULTS.md and
comparison_models/results_summary.csv, next to the reference numbers reported
for YOLOv8n and YOLO11n in docs/yolov8n_vs_yolo11n_comparison.md.

Usage:
    python comparison_models/common/summarize.py
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"YOLOv9t": ROOT / "yolov9", "YOLOv12n": ROOT / "yolov12"}

# Reference values copied from docs/yolov8n_vs_yolo11n_comparison.md (test split,
# full 1,191-image training set, CPU training on Intel i7-1255U).
REFERENCE = {
    "YOLOv8n": {"precision": 0.5288, "recall": 0.6206, "mAP50": 0.6000, "mAP50-95": 0.2914,
                "MAE": 3.2727, "best_epoch": 20, "epochs": "26 (early stop)", "train_h": 2.19,
                "latency": "72.39 ± 16.47 (i7-1255U)", "train_imgs": 1191},
    "YOLO11n": {"precision": 0.5764, "recall": 0.6082, "mAP50": 0.6083, "mAP50-95": 0.2993,
                "MAE": 3.2364, "best_epoch": 30, "epochs": "30", "train_h": 2.41,
                "latency": "67.65 ± 4.77 (i7-1255U)", "train_imgs": 1191},
}


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else (v if v is not None else "n/a")


def main():
    rows = []
    for name, ref in REFERENCE.items():
        rows.append({"model": name, "train_images": ref["train_imgs"], **{k: ref[k] for k in
                     ("precision", "recall", "mAP50", "mAP50-95", "MAE", "best_epoch", "epochs", "train_h",
                      "latency")}, "val_mAP50": None, "val_mAP50-95": None, "source": "reported (docs/)"})

    for name, d in MODELS.items():
        tr = load(d / "metrics" / "train_summary.json") or {}
        te = load(d / "metrics" / "test_metrics.json") or {}
        va = load(d / "metrics" / "val_metrics.json") or {}
        det, cnt, lat = te.get("detection", {}), te.get("counting", {}), te.get("latency", {})
        best = tr.get("best_epoch_val") or {}
        rows.append({
            "model": name,
            "train_images": tr.get("train_images"),
            "precision": det.get("precision"),
            "recall": det.get("recall"),
            "mAP50": det.get("mAP50"),
            "mAP50-95": det.get("mAP50-95"),
            "MAE": cnt.get("MAE"),
            "best_epoch": best.get("epoch"),
            "epochs": tr.get("epochs_completed"),
            "train_h": tr.get("training_duration_h"),
            "latency": f"{lat['inference_ms_mean']} ± {lat['inference_ms_std']} ({lat['hardware']})" if lat else None,
            "val_mAP50": va.get("detection", {}).get("mAP50"),
            "val_mAP50-95": va.get("detection", {}).get("mAP50-95"),
            "source": "measured (this folder)",
        })

    with open(ROOT / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Comparison results: YOLOv8n, YOLO11n, YOLOv9t, YOLOv12n",
        "",
        "> **Not an architecture-only comparison.** YOLOv8n and YOLO11n were trained on the full",
        "> 1,191-image split for up to 30 epochs (about 2.2–2.4 h CPU each). YOLOv9t and YOLOv12n were",
        "> trained on the **594-image half subset** under a **30-minute CPU time budget each**,",
        "> with both models training at the same time on one machine (deadline constraint). As a",
        "> result they completed fewer than 30 epochs. Differences mix architecture, data size,",
        "> training length and hardware.",
        "",
        "## Test split (55 images, 443 trees, unchanged)",
        "",
        "| Model | Train imgs | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Epochs | Best epoch | Train time (h) | CPU inference ms/img | Source |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['train_images']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                     f"{fmt(r['mAP50'])} | {fmt(r['mAP50-95'])} | {fmt(r['MAE'])} | {r['epochs']} | "
                     f"{r['best_epoch']} | {fmt(r['train_h'], 2)} | {r['latency'] or 'n/a'} | {r['source']} |")
    lines += [
        "",
        "## Validation split (116 images, used for checkpoint selection)",
        "",
        "| Model | mAP50 | mAP50-95 |",
        "|---|---|---|",
        *[f"| {r['model']} | {fmt(r['val_mAP50'])} | {fmt(r['val_mAP50-95'])} |" for r in rows if r["val_mAP50"]],
        "",
        "## Notes",
        "",
        "- Tree count = number of predicted boxes at confidence 0.25, the same rule as `src/evaluate_counting.py`.",
        "- Latency: 30 test images × 3 passes after warm-up, batch 1, 416 px, CPU. The new models were measured",
        "  on an Intel i5-12500H and the baselines on an i7-1255U, so compare milliseconds only within a machine.",
        "- The test split was evaluated once per new model, after training. Checkpoint selection used validation only.",
        "  The test split had already been evaluated several times for YOLOv8n and YOLO11n.",
        "- Per-image counts: `yolov9/metrics/test_counting.csv`, `yolov12/metrics/test_counting.csv`.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print((ROOT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
