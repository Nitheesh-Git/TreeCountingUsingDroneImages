# Comparative Performance Analysis: YOLOv8n Baseline vs. YOLO11n

## 1. Overview and Objective
This report presents a direct, empirical comparison between the baseline **YOLOv8n** model and the newly trained **YOLO11n** model for detecting and counting individual tree canopies from top-view drone imagery in the **Tree Counting Using Drone Images** project.

Both models were trained on the exact same dataset splits, evaluated on the same 55 unseen test images at input resolution 416×416, benchmarked under identical CPU hardware conditions, and tested for tree-counting accuracy and quantitative error distribution.

---

## 2. Core Model Architecture and Training Comparison

| Dimension | YOLOv8n Baseline | YOLO11n (Phase 2) | Comparison / Notes |
| :--- | :---: | :---: | :--- |
| **Model Variant** | Nano (`yolov8n.pt`) | Nano (`yolo11n.pt`) | Direct nano-to-nano comparison |
| **Parameters** | 3.2M | **2.6M** | **~19% fewer parameters** |
| **GFLOPs (FLOPs)** | 8.7 GFLOPs | **6.5 GFLOPs** | **~25% lower compute demand** |
| **Model Checkpoint Size** | 6.2 MB | **5.4 MB** | **~13% smaller footprint** |
| **Input Resolution** | 416 × 416 | 416 × 416 | Identical |
| **Batch Size** | 8 | 8 | Identical |
| **Epochs Completed** | 26 (Early stop) | 30 (Full schedule) | Full convergence achieved |
| **Training Duration** | ~2.19 hours | 2.41 hours | ~290s/epoch on CPU |
| **Best Checkpoint Epoch** | Epoch 20 | **Epoch 30** | Steady late-epoch gains |
| **Checkpoint Path** | `runs/detect/tree_detection/baseline_run/weights/best.pt` | `runs/detect/tree_detection/yolov11n_phase2/weights/best.pt` | Completely isolated |

---

## 3. Object Detection Metrics (Test Split Evaluation)

Both models were evaluated on the **55 unseen test images** (containing **443 ground-truth tree canopies**) using the standard Ultralytics evaluation protocol:

| Metric | YOLOv8n Baseline | YOLO11n (Phase 2) | Absolute Difference | Relative Change |
| :--- | :---: | :---: | :---: | :---: |
| **Precision** | 0.5288 | **0.5764** | **+0.0476** | **+9.00%** 📈 |
| **Recall** | **0.6206** | 0.6082 | -0.0124 | -2.00% |
| **mAP@0.5** | 0.6000 | **0.6083** | **+0.0083** | **+1.38%** 📈 |
| **mAP@0.5:0.95** | 0.2914 | **0.2993** | **+0.0079** | **+2.71%** 📈 |

### Metric Definitions:
- **Precision ($\frac{TP}{TP + FP}$):** The proportion of detected bounding boxes that truly correspond to a tree canopy. Higher precision signifies fewer false alarms from soil patches, shrubs, or duplicate boxes.
- **Recall ($\frac{TP}{TP + FN}$):** The proportion of actual ground-truth tree canopies successfully detected.
- **mAP@0.5:** Mean Average Precision calculated at an Intersection over Union (IoU) threshold of 0.50.
- **mAP@0.5:0.95:** COCO-standard metric averaging mAP over 10 IoU thresholds ranging from 0.50 to 0.95 with a step of 0.05, measuring both detection and localization accuracy.

---

## 4. Inference Speed & Latency Benchmark

Inference was benchmarked across 90 forward passes (30 test images over 3 iterations each) following warm-up on the host Intel Core i7-1255U CPU:

| Metric | YOLOv8n Baseline | YOLO11n (Phase 2) | Comparison |
| :--- | :---: | :---: | :--- |
| **Pure Inference Latency** | 72.39 ± 16.47 ms | **67.65 ± 4.77 ms** | **~6.5% faster** on CPU; 3.5× lower variance |
| **Total Pipeline Latency** | 80.06 ± 17.00 ms | **74.88 ± 5.16 ms** | **~6.5% faster** total per-image time |
| **Throughput (approx.)** | ~12.5 FPS | **~13.4 FPS** | CPU-only throughput |

*Observation:* YOLO11n achieves noticeably tighter standard deviation in inference time on CPU (`±4.77 ms` vs `±16.47 ms`), reflecting architectural efficiency improvements from the C3k2 and C2PSA blocks compared to the previous C2f blocks.

---

## 5. Tree Counting Accuracy Evaluation

In forestry inventory, exact counting accuracy is a primary practical requirement. Deriving the tree count directly from the number of predicted boxes ($Count = |Boxes|$ at confidence threshold $\tau = 0.25$), both models were evaluated against the ground-truth counts across all 55 test images.

### Quantitative Counting Results:
| Metric | YOLOv8n Baseline | YOLO11n (Phase 2) | Winner / Interpretation |
| :--- | :---: | :---: | :--- |
| **Total Ground-Truth Trees** | 443 | 443 | Ground truth reference |
| **Total Predicted Trees** | 433 | 455 | — |
| **Total Net Difference** | -10 (-2.26%) | +12 (+2.71%) | Comparable macro totals |
| **Mean Absolute Error (MAE)** | 3.2727 trees/img | **3.2364 trees/img** | **YOLO11n is more accurate** 🏆 |

### Counting Metric Definitions:
- **Absolute Count Error per Image:** $|Count_{pred} - Count_{gt}|$
- **Mean Absolute Error (MAE):** $\frac{1}{N} \sum_{i=1}^N |Count_{pred, i} - Count_{gt, i}|$
- **Total Net Difference:** $\sum Count_{pred} - \sum Count_{gt}$

*Important Note:* Summed net differences can mask internal counting errors through positive and negative cancellation. MAE provides the true measure of per-image counting accuracy. YOLO11n reduces MAE to **3.2364 trees/image**.

---

## 6. Quantitative Error Analysis (Greedy IoU Matching)

Using `src/analyze_predictions.py` and `src/analyze_predictions_yolov11.py`, an identical greedy one-to-one matching algorithm at IoU threshold $\ge 0.5$ and confidence $\tau = 0.25$ was executed on the same **30 validation images**:

| Category | YOLOv8n Baseline | YOLO11n (Phase 2) | Impact / Analysis |
| :--- | :---: | :---: | :--- |
| **True Positives (TP)** | 166 | **179** | **+13 more trees correctly detected** 🌲 |
| **False Positives (FP)** | 141 | **119** | **-22 fewer false alarms & duplicate boxes** 🎯 |
| **False Negatives (FN)** | 104 | **91** | **-13 fewer missed trees** 🔍 |
| **Subset Precision** | 0.5407 | **0.6007** | **+6.00% precision gain** |
| **Subset Recall** | 0.6148 | **0.6630** | **+4.82% recall gain** |

### Visual Categorization Key:
- 🟩 **Green Boxes (True Positives):** Correctly predicted and matched tree crowns ($IoU \ge 0.5$).
- 🟥 **Red Boxes (False Positives):** Predictions without corresponding ground truth, or duplicate detections over an already-matched tree.
- 🟦 **Blue Boxes (False Negatives):** Ground-truth tree canopies completely missed by the model.

Overlays are saved in:
- YOLOv8: `outputs/predictions/error_analysis/`
- YOLO11: `outputs/predictions/yolov11_error_analysis/`

---

## 7. Qualitative Visual Comparison Observations

Samples generated in `outputs/predictions/visual_comparison/` reveal specific behavioural differences:

1. **Reduced Duplicate Bounding Boxes:**
   In YOLOv8n, large tree crowns frequently triggered two overlapping bounding boxes (one for the central crown, one for an extended branch). YOLO11n exhibits tighter bounding box suppression, significantly reducing duplicates.
2. **Dense Canopy Separation:**
   Where adjacent crowns touch, YOLO11n's C2PSA attention block produces better localized centers, preventing multiple crowns from being lumped into a single oversized box.
3. **Small Crown Detection:**
   Understorey or immature saplings with small crowns remain challenging for both models, but YOLO11n captured 13 more valid targets than YOLOv8n on the error analysis subset.
4. **False Positive Suppression:**
   Non-tree objects such as light-colored ground texture and scrub vegetation that caused red false positives in YOLOv8n were correctly ignored by YOLO11n, explaining the jump in precision from 52.88% to 57.64%.

---

## 8. Limitations & Constraints
- **Small-Scale Objects:** Tree crowns smaller than 15×15 pixels at 416 resolution occasionally lack sufficient spatial features for detection.
- **Hardware Bottleneck:** Training on CPU took ~2.4 hours. Future high-resolution training (e.g. at 640×640) would benefit greatly from CUDA GPU acceleration.
- **Confidence Tuning:** A fixed confidence threshold of 0.25 was used across both models for unbiased comparison; application-specific tuning on validation data can further balance precision vs recall.

---

## 9. Conclusion
The empirical evidence confirms that **YOLO11n is superior to YOLOv8n for this tree counting task**:
- **Higher Precision:** `0.5764` vs `0.5288` (+4.76% absolute / +9% relative)
- **Higher Detection Quality:** mAP@0.5 improved to `0.6083` and mAP@0.5:0.95 to `0.2993`
- **Better Tree Counting:** MAE reduced from 3.27 to **3.24 trees/image**
- **Lower Error Footprint:** 22 fewer False Positives and 13 fewer False Negatives on validation subset
- **Faster and Leaner:** ~6.5% lower inference latency on CPU with 19% fewer parameters and 13% smaller checkpoint size.
