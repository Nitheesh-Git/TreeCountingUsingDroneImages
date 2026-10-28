# YOLOv11 Phase 2 — Model Training and Test Evaluation Report

## 1. Executive Summary
This document records the full implementation, training run, validation progression, and final test evaluation of the **YOLO11n** (Nano variant) object detection model for aerial tree crown detection in the **Tree Counting Using Drone Images** project.

Training completed all 30 planned epochs on CPU, establishing consecutive benchmark records across precision, mAP@0.5, and mAP@0.5:0.95. The best checkpoint was saved to `runs/detect/tree_detection/yolov11n_phase2/weights/best.pt`.

---

## 2. Technical Environment & Hardware Specifications
The experiment was conducted on the host CPU environment inside the local virtual environment (`.venv`):

- **Operating System:** Windows 11 Home (64-bit)
- **Processor:** 12th Gen Intel(R) Core(TM) i7-1255U (10 physical cores / 12 logical threads)
- **RAM:** 15.69 GB Total (~3.4 GB available during launch)
- **Accelerator / GPU:** None (`torch.cuda.is_available() == False`)
- **Device:** `cpu`
- **Python:** `3.13.5` (MSC v.1943 64-bit)
- **PyTorch:** `2.14.0+cpu`
- **torchvision:** `0.29.0+cpu`
- **Ultralytics:** `8.4.149` (verified native support for YOLO11 architecture)
- **OpenCV:** `5.0.0`
- **NumPy:** `2.5.3`
- **Polars:** `1.44.2`
- **Matplotlib:** `3.11.2`
- *(Note on reported dependencies: pandas was not present in the environment; Polars was utilized for tabular operations).*

---

## 3. Dataset Verification
The dataset was loaded and validated strictly from the existing configuration file:

- **Dataset Configuration:** `datasets/Tree-Top-View/data.yaml`
- **Source:** Roboflow Universe Tree-Top-View (v2, CC BY 4.0 license)
- **Target Class:** Single class `tree-top` (`id: 0`)
- **Input Dimensions:** 416 × 416 pixels

### Verified Split Distribution:
| Split | Images Verified | Labels Verified | Instances | Image Resolution |
| :--- | :---: | :---: | :---: | :---: |
| **Train** | 1,191 | 1,191 | 8,924 | 416 × 416 |
| **Validation** | 116 | 116 | 932 | 416 × 416 |
| **Test** | 55 | 55 | 443 | 416 × 416 |
| **Total** | **1,362** | **1,362** | **10,299** | 416 × 416 |

*Isolation Notice:* The test dataset (55 images) was kept completely isolated from training, validation, checkpoint selection, and hyperparameter decisions.

---

## 4. Model Architecture & Training Configuration
- **Model:** YOLO11 Nano (`yolo11n.pt`)
- **Pretrained Weights:** Transferred 448/499 items from `yolo11n.pt` base COCO model.
- **Layers & Parameters:** 181 layers, 2,590,035 parameters (fused: 100 layers, 2,582,347 parameters), 6.5 GFLOPs.

### Controlled Training Hyperparameters:
| Parameter | Setting | Rationale / Baseline Consistency |
| :--- | :--- | :--- |
| **Image Size (`imgsz`)** | 416 | Matches YOLOv8n baseline |
| **Batch Size (`batch`)** | 8 | Matches baseline, optimal for 16GB RAM CPU |
| **Epochs** | 30 | Full planned schedule |
| **Patience** | 5 | Early stopping patience |
| **Device** | `cpu` | Host CPU training |
| **Random Seed** | 0 | Deterministic initialization matching baseline |
| **Optimizer** | `AdamW` (auto-selected) | Initial lr=0.002, momentum=0.9, weight decay=0.0005 |
| **DataLoader Workers** | 4 | Adjusted for stable CPU thread scheduling on Windows |
| **Mosaic Augmentation** | Enabled (1.0) | Closes at epoch 20 (`close_mosaic=10`) |
| **Output Directory** | `runs/detect/tree_detection/yolov11n_phase2/` | Independent run directory |

---

## 5. Training Execution & Progression
- **Total Duration:** 02 hours, 25 minutes, 06 seconds (`8,706.9` seconds)
- **Completed Epochs:** 30 / 30
- **Early Stopping Behavior:** Did not prematurely halt; steady improvements through late epochs kept the patience counter active.
- **Average Epoch Time:** ~290 seconds / epoch on CPU.

### Epoch Progression Milestones:
- **Epoch 1:** Initial validation: mAP50 = `0.2795`, mAP50-95 = `0.1178`, Precision = `0.3226`.
- **Epoch 5:** Precision surged to `0.4559`, mAP50 reached `0.4130`.
- **Epoch 10:** First milestone crossing 0.50 mAP: mAP50 = `0.5103`, Recall = `0.5198`.
- **Epoch 15:** mAP50 reached `0.5320`, mAP50-95 reached `0.2637`, Val Class Loss dropped below 1.50 (`1.4914`).
- **Epoch 20:** Major breakthrough: mAP50 = `0.5595`, mAP50-95 = `0.2801`, Recall = `0.5715`.
- **Epoch 21–30 (Mosaic Closed):** Disabling mosaic augmentation enabled fine-tuning on natural non-stitched aerial tree crowns:
  - Epoch 25: mAP50 = `0.5618`, mAP50-95 = `0.2874`, Precision = `0.5271`.
  - Epoch 27: mAP50 = `0.5662`, mAP50-95 = `0.2901`, Precision = `0.5383`.
  - Epoch 28: Precision reached `0.5533`, Val Box Loss reached `1.6269`.
  - Epoch 29: mAP50 peaked at `0.5687`, Val Class Loss dropped to `1.3568`.
  - Epoch 30: Final validation mAP50 = `0.5710`, mAP50-95 = `0.2920`.

### Best Checkpoint:
- **Location:** `runs/detect/tree_detection/yolov11n_phase2/weights/best.pt`
- **File Size:** 5.4 MB
- **Validation Metrics (116 Validation Images):**
  - **Precision:** `0.5550`
  - **Recall:** `0.5760`
  - **mAP@0.5:** `0.5720`
  - **mAP@0.5:0.95:** `0.2920`

---

## 6. Final Test Set Evaluation (Unseen 55 Images)
Evaluated strictly after model training using `src/evaluate_yolov11.py`:

```bash
python src/evaluate_yolov11.py --weights runs/detect/tree_detection/yolov11n_phase2/weights/best.pt --split test --name yolov11n_test_evaluation
```

### Verified Test Split Metrics (55 Images, 443 Ground-Truth Trees):
| Metric | Value |
| :--- | :---: |
| **Precision** | **0.5764** |
| **Recall** | **0.6082** |
| **mAP@0.5** | **0.6083** |
| **mAP@0.5:0.95** | **0.2993** |
| **Inference Latency** | **60.28 ms / image** |
| **Preprocess Latency** | **0.88 ms / image** |
| **Postprocess Latency** | **4.58 ms / image** |
| **Total Latency** | **65.75 ms / image** |

*Output Directory:* `outputs/metrics/yolov11n_test_evaluation/`

---

## 7. Warnings, Resource Constraints & Known Limitations
1. **CPU Execution:** Training required ~2.4 hours due to CPU-only execution without an NVIDIA CUDA GPU.
2. **Dense Canopy Challenges:** Very small, overlapping, or shadowed tree crowns in dense clusters remain the primary source of false negatives.
3. **Artifact Isolation:** Model weights (`best.pt`, `last.pt`, `yolo11n.pt`) and local training directories (`runs/`) are excluded from Git tracking via `.gitignore`.
