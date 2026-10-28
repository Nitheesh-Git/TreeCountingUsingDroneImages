# Detailed Comparison Report: YOLOv9t and YOLOv12n vs YOLOv8n and YOLO11n

**Project:** Tree Counting Using Drone Images
**Dataset:** Tree-Top-View (Roboflow v2, NEON aerial imagery, 1 class `tree-top`, 416 × 416 px)
**Reference models:** YOLOv8n (baseline), YOLO11n (proposed model). Their numbers are taken from
`docs/yolov8n_vs_yolo11n_comparison.md`.
**Comparison models:** YOLOv9t, YOLOv12n (this folder)

> **Important caveat for every table below.** The models were not trained under identical conditions:
>
> | | YOLOv8n / YOLO11n | YOLOv9t / YOLOv12n |
> |---|---|---|
> | Training images | 1,191 (full split) | 594 (half subset) |
> | Training length | up to 30 epochs (26 and 30 completed) | 0.5 h CPU budget each → 17 epochs |
> | Training hardware | Intel i7-1255U, one model at a time | Intel i5-12500H, both models at the same time |
>
> The results therefore show how each model performs **under its own training setup**. They do not isolate
> the effect of architecture alone. The comparison models had less data and fewer epochs, so their
> scores are probably a lower bound on what they could reach.

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

## 2. Detection accuracy (test split: 55 images, 443 trees)

Ultralytics validation protocol: conf 0.001, NMS IoU 0.7, mAP computed over the full precision–recall curve.

| Model | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| YOLOv8n | 0.5288 | **0.6206** | 0.5710 | 0.6000 | 0.2914 |
| YOLO11n | **0.5764** | 0.6082 | **0.5919** | **0.6083** | **0.2993** |
| YOLOv9t | 0.5052 | 0.5532 | 0.5281 | 0.5256 | 0.2640 |
| YOLOv12n | 0.5219 | 0.5421 | 0.5318 | 0.5383 | 0.2596 |

F1 = 2PR / (P + R). For YOLOv8n and YOLO11n it is computed from the reported P and R.

**Validation split (116 images), used to pick the best checkpoint:**

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Count MAE |
|---|---|---|---|---|---|
| YOLOv9t | 0.5294 | 0.4839 | 0.5188 | 0.2719 | 3.40 |
| YOLOv12n | 0.5309 | 0.5440 | 0.5525 | 0.2761 | 2.99 |

**Observations**
- YOLO11n is the best model on every detection metric except recall, where YOLOv8n is highest.
- On test mAP@0.5, YOLOv12n trails YOLO11n by 0.07 and YOLOv9t by 0.08. Both comparison models had
  half the data and about 57% of the epochs.
- Between the comparison models, YOLOv12n is ahead on precision, F1 and mAP@0.5. YOLOv9t is
  slightly ahead on recall and mAP@0.5:0.95. On validation, YOLOv12n is ahead on every metric.

---

## 3. Counting accuracy (test split)

Tree count = number of predicted boxes at confidence ≥ 0.25, the same rule as `src/evaluate_counting.py`.

| Model | Predicted / GT trees | Net difference | **MAE** (trees/img) | RMSE | Bias (mean signed error) |
|---|---|---|---|---|---|
| YOLOv8n | 433 / 443 | −10 (−2.3%) | 3.27 | n/a | −0.18 |
| YOLO11n | 455 / 443 | +12 (+2.7%) | 3.24 | n/a | +0.22 |
| YOLOv9t | 428 / 443 | −15 (−3.4%) | **2.85** | 4.04 | −0.27 |
| YOLOv12n | 394 / 443 | −49 (−11.1%) | 3.04 | **4.02** | −0.89 |

Baseline bias is the reported net difference divided by 55. The baselines did not report RMSE.

**Count accuracy per image (comparison models)**

| Model | Exact count | Within ±1 | Within ±2 | Over-counted images | Under-counted images | Mean relative error |
|---|---|---|---|---|---|---|
| YOLOv9t | 16.4% | 40.0% | 58.2% | 21 | 25 | 39.3% |
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
- YOLOv9t has the **lowest counting MAE of all four models** (2.85). It is better than YOLO11n by 0.38
  trees per image, even though its mAP is lower.
- Counting MAE and mAP do not always rank models the same way. MAE only checks how many boxes there
  are, not where they are. Over-counting in one image and under-counting in another can each still
  produce a count close to the truth.
- YOLOv12n systematically **under-counts** (−49 trees, bias −0.89). It is the most conservative of the four.
- Error grows with density for both models: dense canopies (11+ trees) roughly triple the MAE of sparse
  plots. DEJU (dense boreal forest, Alaska) is the hardest site for YOLOv9t.
- MAE by site is based on very few images per site (1–10), so treat it as an indication only.

---

## 4. Error analysis: TP / FP / FN

The same algorithm as `src/analyze_predictions_yolov11.py`: predictions sorted by confidence and matched
one-to-one to ground truth at IoU ≥ 0.5, conf ≥ 0.25.

**First 30 sorted validation images (the same subset used for YOLOv8n and YOLO11n)**

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| YOLOv8n | 166 | 141 | 104 | 0.5407 | 0.6148 | 0.5754 |
| YOLO11n | **179** | 119 | **91** | **0.6007** | **0.6630** | **0.6303** |
| YOLOv9t | 152 | 117 | 118 | 0.5651 | 0.5630 | 0.5640 |
| YOLOv12n | 159 | **109** | 111 | 0.5933 | 0.5889 | 0.5911 |

**Full test split (55 images, 443 trees). Comparison models only; this was not computed for the baselines.**

| Model | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| YOLOv9t | 226 | 202 | 217 | 0.5280 | 0.5102 | 0.5189 |
| YOLOv12n | 224 | **170** | 219 | **0.5685** | 0.5056 | **0.5352** |

**Observations**
- **YOLOv12n has the fewest false positives of all four models** on the 30-image subset (109). On the
  test split it has 32 fewer false alarms than YOLOv9t while finding almost the same number of trees.
  This fits its attention-based design, which suppresses background such as soil and shrubs.
- Both comparison models have more **false negatives** (missed trees) than YOLO11n. That is the main
  reason for their lower recall and mAP, and it is what you'd expect from less training data and fewer epochs.
- In counting terms, YOLOv9t's extra false positives partly cancel out its misses, which is why its count
  MAE is lower even though its box-level precision is lower.

---

## 5. Model efficiency

| Model | Parameters | GFLOPs (640 px) | GFLOPs (416 px) | Checkpoint (best.pt) | CPU inference latency |
|---|---|---|---|---|---|
| YOLOv8n | 3.2 M | 8.7 | n/a | 6.2 MB | 72.4 ± 16.5 ms (i7-1255U) |
| YOLO11n | 2.6 M | 6.5 | n/a | 5.4 MB | 67.7 ± 4.8 ms (i7-1255U) |
| YOLOv9t | **2.11 M** | 8.37 | 3.54 | **4.82 MB** | 21.5 ± 0.5 ms (i5-12500H) |
| YOLOv12n | 2.58 M | 7.53 | **2.94** | 5.52 MB | **21.2 ± 0.6 ms** (i5-12500H) |

Latency protocol: 30 test images × 3 passes after warm-up, batch 1, 416 px, PyTorch on CPU. The baselines'
GFLOPs are the reported Ultralytics nominal values. The comparison models' GFLOPs were measured on the trained,
single-class models with the Ultralytics profiler.

**Observations**
- **YOLOv9t is the smallest model**: 34% fewer parameters than YOLOv8n and 19% fewer than YOLO11n, with
  the smallest checkpoint. It has many small layers (918 modules, against 466 for YOLOv12n), so the low
  parameter count does not lead to fewer FLOPs.
- **YOLOv12n needs the least compute at 416 px** (2.94 GFLOPs, against 3.54 for YOLOv9t).
- On the same CPU, YOLOv9t and YOLOv12n have practically the same latency (about 21 ms, roughly 47 FPS).
- **The latency numbers cannot be compared across the two machines.** The baselines were timed on a
  different CPU and software stack. The roughly 3× gap reflects the hardware, not the models. A fair latency
  comparison needs all four checkpoints timed on one machine.

---

## 6. Training cost and convergence

| Model | Training images | Epochs completed | Best epoch | Training time | Time per epoch | Stopped by |
|---|---|---|---|---|---|---|
| YOLOv8n | 1,191 | 26 | 21 | 2.19 h | ≈ 300 s | early stopping (patience 5) |
| YOLO11n | 1,191 | 30 | 30 | 2.41 h | ≈ 290 s | max epochs |
| YOLOv9t | 594 | 17 | 17 | 0.51 h | ≈ 107 s* | 0.5 h time budget |
| YOLOv12n | 594 | 17 | 16 | 0.50 h | ≈ 105 s* | 0.5 h time budget |

\* Both models trained at the same time with 8 CPU threads each. Trained alone, YOLOv9t took ≈ 52 s per epoch.

**Validation mAP@0.5 by epoch**

| Epoch | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv9t | 0.207 | 0.294 | 0.435 | 0.436 | 0.409 | 0.497 | 0.438 | 0.506 | 0.526 | **0.519**† |
| YOLOv12n | 0.269 | 0.246 | 0.414 | 0.383 | 0.474 | 0.496 | 0.495 | 0.532 | **0.553** | 0.551 |

† YOLOv9t's best checkpoint is epoch 17: its mAP50-95 (0.2720) is the highest, although epoch 16 had the higher mAP50.
For reference, YOLO11n on the full data reached 0.413 at epoch 5, 0.510 at epoch 10, and 0.571 at epoch 30.

**Observations**
- Both comparison models were **still improving** when the time budget stopped them. The best epoch was the
  last one (YOLOv9t, 17) or the second to last (YOLOv12n, 16), and the best checkpoint is chosen by mAP50-95,
  Ultralytics' fitness. With the full 30 epochs they would very likely score higher.
- Despite having half the data, both learned at a similar rate to YOLO11n in the early epochs. Both curves
  fluctuate from epoch to epoch. YOLOv12n is ahead in most epochs from epoch 9 onward and in each of the last three.
- Under the time budget, Ultralytics re-planned each run to 17–18 epochs, so `close_mosaic=10` turned
  mosaic augmentation off after epoch 8 (YOLOv9t) and after epoch 7 (YOLOv12n). Mosaic was therefore off for
  more than half of each run, against the last 10 of 30 epochs for the baselines.
- Training both models together cost about 1 h of CPU time in total, against about 4.6 h for the two
  baselines (with twice the data and more epochs).

---

## 7. Summary: which model wins on what

| Criterion | Best model |
|---|---|
| Detection accuracy (mAP@0.5, mAP@0.5:0.95, precision, F1) | **YOLO11n** |
| Recall (fewest missed trees) | **YOLOv8n** (test) / **YOLO11n** (30-image analysis) |
| Counting accuracy (MAE) | **YOLOv9t** (2.85) |
| Fewest false alarms (FP) | **YOLOv12n** |
| Smallest model (parameters, file size) | **YOLOv9t** |
| Lowest compute (GFLOPs) | **YOLOv12n** at 416 px |
| Best of the two comparison models overall | **YOLOv12n** on detection and validation; **YOLOv9t** on counting |

**Conclusion.**
- The proposed YOLO11n stays the strongest detector. It leads on mAP@0.5 (0.608), mAP@0.5:0.95 (0.299),
  precision and F1.
- The newer architectures show specific strengths:
  - YOLOv12n's attention design gives the fewest false positives and the lowest compute.
  - YOLOv9t's GELAN design gives the smallest model and the best per-image counting error.
- These comparison results come from half the data and about 57% of the epochs, so they are a conservative
  estimate. A fully controlled comparison would retrain all four models on the same data, for the same
  number of epochs, on the same machine.

---

## 8. Limitations

1. **Training data, epochs and hardware differ** between the reference models and the comparison models (see the caveat at the top).
2. **The test set has been used before.** The 55-image test split had already been evaluated several times for the baselines. For the comparison models it was evaluated once, after training, and checkpoint selection used validation only.
3. **Latency** was measured on different CPUs for the two groups of models.
4. **Small test set.** 55 images (443 trees); per-site and per-density results rely on very few images.
5. **The dataset is augmented.** Roboflow created 3 augmented copies of each training image. A few training images share a source name with validation or test images. These were excluded from the half subset, but the baselines' full training set includes them.
6. **A fixed confidence of 0.25** was used for counting across all models. A per-model threshold tuned on validation could change the counting MAE.

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
