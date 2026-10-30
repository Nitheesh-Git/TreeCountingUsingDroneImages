# Comparison results: YOLOv9t vs YOLOv12n

> Both models were trained under identical conditions: the **594-image half subset** (seed 0),
> a **30-minute CPU time budget each** (17 epochs, both trained at the same time), 416 px, batch 8.
> They were evaluated on the original, unchanged validation and test splits. Full details are in [REPORT.md](REPORT.md).

## Test split (55 images, 443 trees, unchanged)

| Model | Train imgs | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Epochs | Best epoch | Train time (h) | CPU inference ms/img |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv9t | 594 | 0.5052 | 0.5532 | 0.5256 | 0.2640 | 2.8545 | 17 | 17 | 0.51 | 21.5 ± 0.45 (12th Gen Intel(R) Core(TM) i5-12500H) |
| YOLOv12n | 594 | 0.5219 | 0.5421 | 0.5383 | 0.2596 | 3.0364 | 17 | 16 | 0.50 | 21.19 ± 0.58 (12th Gen Intel(R) Core(TM) i5-12500H) |

## Validation split (116 images, used for checkpoint selection)

| Model | mAP50 | mAP50-95 |
|---|---|---|
| YOLOv9t | 0.5188 | 0.2719 |
| YOLOv12n | 0.5525 | 0.2761 |

## Notes

- Tree count = number of predicted boxes at confidence 0.25, the same rule as `src/evaluate_counting.py`.
- Latency: 30 test images × 3 passes after warm-up, batch 1, 416 px, Intel i5-12500H CPU.
- The test split was evaluated once per model, after training. Checkpoint selection used validation only.
- Per-image counts: `yolov9/metrics/test_counting.csv`, `yolov12/metrics/test_counting.csv`.
