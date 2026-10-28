# Comparison results: YOLOv8n, YOLO11n, YOLOv9t, YOLOv12n

> **Not an architecture-only comparison.** YOLOv8n and YOLO11n were trained on the full
> 1,191-image split for up to 30 epochs (about 2.2–2.4 h CPU each). YOLOv9t and YOLOv12n were
> trained on the **594-image half subset** under a **30-minute CPU time budget each**,
> with both models training at the same time on one machine (deadline constraint). As a
> result they completed fewer than 30 epochs. Differences mix architecture, data size,
> training length and hardware.

## Test split (55 images, 443 trees, unchanged)

| Model | Train imgs | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Epochs | Best epoch | Train time (h) | CPU inference ms/img | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv8n | 1191 | 0.5288 | 0.6206 | 0.6000 | 0.2914 | 3.2727 | 26 (early stop) | 21 | 2.19 | 72.39 ± 16.47 (i7-1255U) | reported (docs/) |
| YOLO11n | 1191 | 0.5764 | 0.6082 | 0.6083 | 0.2993 | 3.2364 | 30 | 30 | 2.41 | 67.65 ± 4.77 (i7-1255U) | reported (docs/) |
| YOLOv9t | 594 | 0.5052 | 0.5532 | 0.5256 | 0.2640 | 2.8545 | 17 | 17 | 0.51 | 21.5 ± 0.45 (12th Gen Intel(R) Core(TM) i5-12500H) | measured (this folder) |
| YOLOv12n | 594 | 0.5219 | 0.5421 | 0.5383 | 0.2596 | 3.0364 | 17 | 16 | 0.50 | 21.19 ± 0.58 (12th Gen Intel(R) Core(TM) i5-12500H) | measured (this folder) |

## Validation split (116 images, used for checkpoint selection)

| Model | mAP50 | mAP50-95 |
|---|---|---|
| YOLOv9t | 0.5188 | 0.2719 |
| YOLOv12n | 0.5525 | 0.2761 |

## Notes

- Tree count = number of predicted boxes at confidence 0.25, the same rule as `src/evaluate_counting.py`.
- Latency: 30 test images × 3 passes after warm-up, batch 1, 416 px, CPU. The new models were measured
  on an Intel i5-12500H and the baselines on an i7-1255U, so compare milliseconds only within a machine.
- The test split was evaluated once per new model, after training. Checkpoint selection used validation only.
  The test split had already been evaluated several times for YOLOv8n and YOLO11n.
- Per-image counts: `yolov9/metrics/test_counting.csv`, `yolov12/metrics/test_counting.csv`.
