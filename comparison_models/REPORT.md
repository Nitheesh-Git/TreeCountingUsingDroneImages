# Detailed Comparison Report: YOLOv9t vs YOLOv12n

**Project:** Tree Counting Using Drone Images
**Dataset:** Tree-Top-View (Roboflow v2, NEON aerial imagery, 1 class `tree-top`, 416 × 416 px)
**Models:** YOLOv9t (`yolov9t.pt`) and YOLOv12n (`yolo12n.pt`), both COCO-pretrained and fine-tuned on the dataset

> **Training setup (applies to every table below).** Both models were trained under identical conditions, so
> the comparison between them is controlled:
>
> | Setting | YOLOv9t and YOLOv12n |
> |---|---|
> | Training images | 594 (reproducible half subset of the 1,191-image train split, seed 0) |
> | Validation / test | original, unchanged 116 / 55 images |
> | Training length | 0.5 h CPU time budget each → 17 epochs (deadline constraint) |
> | Hardware | Intel i5-12500H CPU; both models trained at the same time, 8 threads each |
> | Settings | 416 px, batch 8, seed 0, optimizer auto (AdamW lr 0.002), same augmentation |
>
> Because each model had only half the training data and 17 epochs, these scores are a conservative
> estimate of what the two architectures can reach.

---

## 1. Evaluation parameters

The report uses five groups of parameters:

| # | Parameter group | Metrics | Why it matters for tree counting |
|---|---|---|---|
| 1 | **Detection accuracy** | Precision, Recall, F1, mAP@0.5, mAP@0.5:0.95 | Whether the model finds tree crowns and places the boxes correctly |
| 2 | **Counting accuracy** | MAE, RMSE, bias, % of images with an exact count or within ±1 / ±2 | The project's actual goal: the number of trees per image |
| 3 | **Error analysis** | True positives (TP), false positives (FP), false negatives (FN), by greedy IoU matching | *Why* counts go wrong: missed trees vs false alarms |
| 4 | **Model efficiency** | Parameters, GFLOPs, checkpoint size, CPU inference latency | Whether the model can be deployed on a drone, edge device or laptop |
| 5 | **Training cost and convergence** | Training time, epochs, best epoch, learning curve | How much compute each model needs and whether it had converged |

---

## 2. Detection accuracy

Ultralytics validation protocol: conf 0.001, NMS IoU 0.7, mAP computed over the full precision–recall curve.
F1 = 2PR / (P + R).

**Test split (55 images, 443 trees)**

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv9t | 0.5052 | **0.5532** | 0.5281 | 0.5256 | **0.2640** |
| YOLOv12n | **0.5219** | 0.5421 | **0.5318** | **0.5383** | 0.2596 |

**Validation split (116 images, 932 trees), used to pick the best checkpoint**

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv9t | 0.5294 | 0.4839 | 0.5056 | 0.5188 | 0.2719 |
| YOLOv12n | **0.5309** | **0.5440** | **0.5374** | **0.5525** | **0.2761** |

**Observations**
- **YOLOv12n is the stronger detector overall.** It is ahead on every validation metric. On the test split it is
  ahead on precision, F1 and mAP@0.5.
- YOLOv9t is slightly ahead on test recall (+0.011) and test mAP@0.5:0.95 (+0.004). Both margins are small
  for a 55-image test set.
- The gap is larger on validation (mAP@0.5 +0.034 for YOLOv12n) than on test (+0.013).

---

## 3. Counting accuracy (test split)

Tree count = number of predicted boxes at confidence ≥ 0.25, the same rule as `src/evaluate_counting.py`.

| Model | Predicted / GT trees | Net difference | **MAE** (trees/img) | RMSE | Bias (mean signed error) |
|---|---|---|---|---|---|
| YOLOv9t | 428 / 443 | −15 (−3.4%) | **2.85** | 4.04 | **−0.27** |
| YOLOv12n | 394 / 443 | −49 (−11.1%) | 3.04 | **4.02** | −0.89 |

Validation-split MAE: YOLOv9t 3.40, YOLOv12n 2.99.

**Count accuracy per image**

| Model | Exact count | Within ±1 | Within ±2 | Over-counted images | Under-counted images | Mean relative error |
|---|---|---|---|---|---|---|
| YOLOv9t | 16.4% | **40.0%** | **58.2%** | 21 | 25 | **39.3%** |
| YOLOv12n | 16.4% | 38.2% | 52.7% | 20 | 26 | 45.9% |

**MAE by tree density (trees per image in ground truth)**

| Density | Images | YOLOv9t MAE | YOLOv12n MAE |
|---|---|---|---|
| 0–5 trees | 20 | **1.50** | 1.90 |
| 6–10 trees | 19 | **2.63** | 3.00 |
| 11+ trees | 16 | 4.81 | **4.50** |

**MAE by NEON site (test images per site in brackets)**

| Site | YOLOv9t | YOLOv12n | Site | YOLOv9t | YOLOv12n |
|---|---|---|---|---|---|
| ABBY (8) | 2.50 | 3.25 | GUAN (2) | 1.00 | 1.00 |
| BART (10) | 1.70 | 3.70 | HEAL (4) | 1.25 | 2.25 |
| BONA (1) | 4.00 | 5.00 | JERC (2) | 3.00 | 3.50 |
| CLBJ (2) | 3.50 | 2.00 | SJER (5) | 0.60 | 1.20 |
| DEJU (7) | 6.57 | 4.00 | TEAK (4) | 4.00 | 3.75 |
| DELA (3) | 4.33 | 3.67 | DSNY (6) | 2.67 | 2.67 |
| GRSM (1) | 2.00 | 1.00 | | | |

**Observations**
- **YOLOv9t counts trees more accurately on the test split.** It has the lower MAE (2.85 vs 3.04), the smaller
  total error (−15 vs −49 trees), and more images within ±1 and ±2 trees.
- YOLOv9t has lower mAP but better counts. The per-image count error equals |FP − FN|, so false alarms and
  missed trees in the same image cancel out. YOLOv9t makes more false alarms (section 4), and these offset
  its missed trees.
- YOLOv12n systematically **under-counts** (bias −0.89 trees per image).
- Error grows with density for both models: dense canopies (11+ trees) roughly triple the MAE of sparse
  plots. YOLOv9t is better on sparse and medium plots, and YOLOv12n on dense ones. DEJU (dense boreal
  forest, Alaska) is the hardest site for YOLOv9t.
- On the validation split the order reverses (YOLOv12n 2.99 vs YOLOv9t 3.40). The counting advantage is not
  consistent across splits.
- MAE by site is based on very few images per site (1–10), so treat it as an indication only.

---

## 4. Error analysis: TP / FP / FN

Predictions are sorted by confidence and matched one-to-one to ground truth at IoU ≥ 0.5, conf ≥ 0.25. This is
the same greedy algorithm as the project's existing `src/analyze_predictions*.py` scripts.

**First 30 sorted validation images (270 trees), the project's standard error-analysis subset**

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| YOLOv9t | 152 | 117 | 118 | 0.5651 | 0.5630 | 0.5640 |
| YOLOv12n | **159** | **109** | **111** | **0.5933** | **0.5889** | **0.5911** |

**Full test split (55 images, 443 trees)**

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| YOLOv9t | **226** | 202 | **217** | 0.5280 | **0.5102** | 0.5189 |
| YOLOv12n | 224 | **170** | 219 | **0.5685** | 0.5056 | **0.5352** |

**Observations**
- **YOLOv12n makes far fewer false alarms.** On the test split it has 32 fewer false positives (170 vs 202)
  and finds almost the same number of real trees (224 vs 226). This fits its attention-based design, which
  helps it suppress background such as soil, shadows and shrubs.
- On the 30-image validation subset, YOLOv12n is better on all three counts: more TP, fewer FP and fewer FN.
- Both models miss about half of the trees at conf 0.25 (FN ≈ 217–219 of 443). Missed trees are the main
  source of error for both. That is expected with half the training data and 17 epochs.

---

## 5. Model efficiency

| Model | Parameters | GFLOPs (640 px) | GFLOPs (416 px) | Checkpoint (best.pt) | CPU latency, pure inference | CPU latency, total |
|---|---|---|---|---|---|---|
| YOLOv9t | **2.11 M** | 8.37 | 3.54 | **4.82 MB** | 21.50 ± 0.45 ms | 22.20 ± 0.47 ms |
| YOLOv12n | 2.58 M | **7.53** | **2.94** | 5.52 MB | **21.19 ± 0.58 ms** | **21.88 ± 0.58 ms** |

Latency protocol: 30 test images × 3 passes after 5 warm-up runs, batch 1, 416 px, PyTorch on the
i5-12500H CPU, with no other job running. GFLOPs were measured on the trained single-class models with the
Ultralytics profiler.

**Observations**
- **YOLOv9t is the smaller model:** 18% fewer parameters and a 13% smaller checkpoint.
- **YOLOv12n needs less compute:** 17% fewer GFLOPs at 416 px. YOLOv9t has many small layers (918 modules
  against 466), so its low parameter count does not lead to fewer FLOPs.
- **Speed is effectively tied.** Both run at about 21–22 ms per image, roughly 45–47 FPS on a laptop CPU.
  YOLOv12n is 0.3 ms faster, which is within the measurement noise.

---

## 6. Training cost and convergence

| Model | Training images | Epochs completed | Best epoch | Training time | Time per epoch | Stopped by |
|---|---|---|---|---|---|---|
| YOLOv9t | 594 | 17 | 17 | 0.51 h | ≈ 107 s* | 0.5 h time budget |
| YOLOv12n | 594 | 17 | 16 | 0.50 h | ≈ 105 s* | 0.5 h time budget |

\* Both models trained at the same time with 8 CPU threads each. Trained alone, YOLOv9t took ≈ 52 s per epoch.
The best checkpoint is the epoch with the highest validation mAP@0.5:0.95, which is Ultralytics' fitness.

**Validation mAP@0.5 by epoch**

| Epoch | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv9t | 0.207 | 0.294 | 0.435 | 0.436 | 0.409 | 0.497 | 0.438 | 0.506 | 0.526 | **0.519**† |
| YOLOv12n | 0.269 | 0.246 | 0.414 | 0.383 | 0.474 | 0.496 | 0.495 | 0.532 | **0.553** | 0.551 |

† YOLOv9t's best checkpoint is epoch 17: its mAP@0.5:0.95 (0.2720) is the highest, although epoch 16 had the higher mAP@0.5.

**Observations**
- Both models were **still improving** when the time budget stopped them. The best epoch was the last one
  (YOLOv9t) or the second to last (YOLOv12n). With the full 30 epochs both would very likely score higher.
- YOLOv12n starts faster (0.269 vs 0.207 after epoch 1). Both curves fluctuate from epoch to epoch.
  YOLOv12n is ahead in most epochs from epoch 9 onward and in each of the last three.
- Under the time budget, Ultralytics re-planned each run to 17–18 epochs, so `close_mosaic=10` turned mosaic
  augmentation off after epoch 8 (YOLOv9t) and after epoch 7 (YOLOv12n).
- Both models cost the same to train: about 30 minutes each, about 1 hour of CPU time in total.

---

## 7. Summary: which model wins on what

| Criterion | Better model |
|---|---|
| Detection accuracy (mAP@0.5, precision, F1; all validation metrics) | **YOLOv12n** |
| mAP@0.5:0.95 and recall on the test split | **YOLOv9t** (by a small margin) |
| Counting accuracy on the test split (MAE, total count, within ±1/±2) | **YOLOv9t** |
| Counting accuracy on the validation split (MAE) | **YOLOv12n** |
| Fewest false alarms (FP) | **YOLOv12n** |
| Dense plots (11+ trees) | **YOLOv12n** |
| Smallest model (parameters, file size) | **YOLOv9t** |
| Lowest compute (GFLOPs) | **YOLOv12n** |
| CPU speed | tie (≈ 21 ms per image) |
| Training cost | tie (0.5 h each) |

**Conclusion.**
- **YOLOv12n** is the better detector. It is more precise, raises fewer false alarms, handles dense canopies
  better and is ahead on every validation metric, with slightly less compute. Its main weakness is a
  systematic under-count.
- **YOLOv9t** is the smaller model, and it gave the more accurate tree counts on the test split: lower MAE
  and a total count within 3.4% of the truth. Part of that advantage comes from its false positives
  cancelling its misses, and the advantage does not hold on the validation split.
- Both were trained on half the data for 17 epochs, so their absolute scores are conservative.

---

## 8. Limitations

1. **Reduced training.** 594 of 1,191 training images and 17 epochs, imposed by a 0.5 h CPU time budget per model.
2. **Small test set.** 55 images (443 trees); per-site and per-density results rely on very few images. The test split was evaluated once per model after training, and checkpoint selection used validation only.
3. **The dataset is augmented.** Roboflow created 3 augmented copies of each training image. The subset keeps each image's copies together, and leaves out 10 training scenes whose names also appear in validation or test.
4. **A fixed confidence of 0.25** was used for counting. A per-model threshold tuned on validation could change the counting MAE.
5. **Latency is machine-specific.** It was measured on one laptop CPU (i5-12500H). Absolute milliseconds will differ on other hardware.

---

## 9. Reproducibility

| Item | Value |
|---|---|
| Subset | `comparison_models/common/subset_manifest.txt` (594 images, seed 0); statistics in `subset_stats.md` |
| Software | Python 3.12.13, ultralytics 8.4.177, torch 2.14.1+cpu |
| Pretrained weights | `yolov9t.pt`, `yolo12n.pt` (official Ultralytics assets, COCO) |
| Settings | imgsz 416, batch 8, patience 5, seed 0, optimizer auto → AdamW lr 0.002, time 0.5 h |
| Raw metrics | `yolov9/metrics/`, `yolov12/metrics/` (`*_metrics.json`, `report_details.json`, per-image `*_counting.csv`) |

```bash
python comparison_models/yolov9/train.py --time 0.5      # same for yolov12
python comparison_models/yolov9/evaluate.py --split val
python comparison_models/yolov9/evaluate.py --split test
python comparison_models/common/report_details.py        # sections 4–6 of this report
python comparison_models/common/summarize.py             # RESULTS.md
```
