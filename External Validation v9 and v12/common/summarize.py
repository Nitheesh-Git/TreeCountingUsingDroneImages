"""
Collect the comparison-model metrics into External Validation v9 and v12/RESULTS.md and
External Validation v9 and v12/results_summary.csv.

Usage:
    python "External Validation v9 and v12/common/summarize.py"
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # "External Validation v9 and v12/"
MODELS = {"YOLOv9t": ROOT / "yolov9", "YOLOv12n": ROOT / "yolov12"}

def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else (v if v is not None else "n/a")


def main():
    rows = []
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
        })

    with open(ROOT / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "# Comparison results: YOLOv9t vs YOLOv12n",
        "",
        "> Both models were trained under identical conditions: the **594-image half subset** (seed 0),",
        "> a **30-minute CPU time budget each** (17 epochs, both trained at the same time), 416 px, batch 8.",
        "> They were evaluated on the original, unchanged validation and test splits. Full details are in [REPORT.md](REPORT.md).",
        "",
        "## Test split (55 images, 443 trees, unchanged)",
        "",
        "| Model | Train imgs | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Epochs | Best epoch | Train time (h) | CPU inference ms/img |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | {r['train_images']} | {fmt(r['precision'])} | {fmt(r['recall'])} | "
                     f"{fmt(r['mAP50'])} | {fmt(r['mAP50-95'])} | {fmt(r['MAE'])} | {r['epochs']} | "
                     f"{r['best_epoch']} | {fmt(r['train_h'], 2)} | {r['latency'] or 'n/a'} |")
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
        "- Latency: 30 test images × 3 passes after warm-up, batch 1, 416 px, Intel i5-12500H CPU.",
        "- The test split was evaluated once per model, after training. Checkpoint selection used validation only.",
        "- Per-image counts: `yolov9/metrics/test_counting.csv`, `yolov12/metrics/test_counting.csv`.",
        "",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))
    print((ROOT / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
