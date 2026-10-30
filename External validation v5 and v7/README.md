# 🌳 Tree Counting Using Drone Images

An end-to-end aerial computer vision and ecological monitoring system for automated individual tree crown detection, localization, and counting from top-down unmanned aerial vehicle (UAV / drone) RGB imagery using **Ultralytics YOLOv11**, **YOLOv8n**, and external validation benchmarks with **YOLOv5n** and **YOLOv7-tiny**.

---

## 📌 Project Overview

Manual tree counting in forestry, plantation management, and environmental conservation is labor-intensive, slow, and prone to significant observer bias. This project builds a fully automated, reproducible deep learning pipeline that:
1. Detects individual tree crowns from high-resolution drone orthomosaics and aerial tiles.
2. Generates automated per-image and survey-level tree counts.
3. Integrates **YOLOv11n** directly into the core pipeline (`src/`) alongside the **YOLOv8n** baseline.
4. Provides a dedicated external validation & lightweight benchmarking suite (`External validation v5 and v7/`) evaluating **YOLOv5n** and **YOLOv7-tiny** on both a 50% fair CPU training subset (`595` images) and an independent 100-tile neotropical rainforest benchmark (**CanopyRS / SelvaBox** across Ecuador, Brazil, and Panama).

---

## 🎯 Objectives

- **Automated Detection & Counting:** Detect tree crowns and produce accurate counts with near-zero systematic bias.
- **Core YOLOv11 Upgrade:** Train, evaluate, and analyze **YOLO11n** (Phase 1 & Phase 2) directly in the core pipeline against the **YOLOv8n** baseline on the full training split (`1,191` images).
- **External Validation (`v5` & `v7-tiny`):** Benchmark lightweight models (**YOLOv5n** and **YOLOv7-tiny**) under a standardized 50% CPU-friendly training subset (`595` images) inside `External validation v5 and v7/`.
- **Zero-Shot Domain Shift Analysis:** Quantify model transferability across all four trained detectors (`YOLO11n`, `YOLOv8n`, `YOLOv5n`, `YOLOv7-tiny`) on an independent external 100-image dataset (`8,111` ground-truth trees) across variations in canopy density, crown diameter, and image resolution.

---

## 🛠 Technology Stack

| Category | Technology | Description |
| :--- | :--- | :--- |
| **Language & Runtime** | Python 3.13.9 | Modern CPython runtime |
| **Deep Learning Framework** | PyTorch 2.10.0+cpu, Torchvision 0.25.0+cpu | CPU-optimized tensor computation |
| **Primary Vision Engine** | Ultralytics 8.4.21 | Unified YOLO11, YOLOv8, and YOLOv5nu pipeline |
| **Legacy Architecture Engine** | WongKinYiu YOLOv7 (Patched) | Py3.13 / PyTorch 2.10 compatible anchor-based E-ELAN |
| **Computer Vision** | OpenCV 4.13, Pillow | Image preprocessing, annotation rendering, overlays |
| **Data & Scientific Stack** | NumPy 2.4, SciPy 1.17, Pandas 3.0, Polars 1.39 | Metric computation, IoU matching, data processing |
| **External Data Streaming** | aiohttp, pyarrow, fsspec | Chunked asynchronous parquet stream from Hugging Face |
| **Version Control** | Git & GitHub | Source tracking and experimental versioning |

---

## 📂 Datasets

### 1. In-Domain Dataset: `Tree-Top-View`
- **Source:** Roboflow Universe (`tree-top-view`), CC BY 4.0 license.
- **Task:** Single-class object detection (`0: tree-top`).
- **Input Resolution:** `416 × 416` pixels, RGB.
- **Splits:**
  - **Train Split:** `1,191` images, `10,002` ground-truth trees (mean `8.40` trees/image, range `0–62`).
  - **Validation Split:** `149` images, `1,318` ground-truth trees (mean `8.85` trees/image, range `0–55`).
  - **Held-Out Test Split:** `55` images, `443` ground-truth trees (mean `8.05` trees/image, median `8.0`, range `0–25`). Mean crown diameter: `60.7 × 59.2` pixels.
- **Fair CPU Comparison Subset (50%):**
  - Deterministically sampled (`seed=42`, `External validation v5 and v7/prepare_subset.py`): **`595` train images (`5,127` GT trees)**, preserving identical full validation (`149` images) and test (`55` images) splits.

### 2. External Generalization Benchmark: `CanopyRS / SelvaBox` (`testdataset100`)
- **Source:** Hugging Face `CanopyRS/SelvaBox` (official held-out test split, tiles extracted via `External validation v5 and v7/prepare_testdataset100.py`).
- **Scale:** `100` orthomosaic PNG tiles (`1777 × 1777` pixels, `4.5 cm/px` GSD, `80 m × 80 m` footprint) containing **`8,111` ground-truth tree crowns** (mean `81.11` trees/image, range `3–361`).
- **Three Neotropical Ecological Strata:**
  1. **Ecuador (`ecuador_tiputini` - 60 tiles, 1,095 trees):** Moderate density (mean `18.25` trees/image). Crown scale closest to internal dataset.
  2. **Brazil (`brazil_zf2` - 13 tiles, 1,641 trees):** Dense Amazonian closed canopy (mean `126.23` trees/image; `15.7×` denser than internal test).
  3. **Panama (`panama_aguasalud` - 27 tiles, 5,375 trees):** Hyper-dense multi-tier neotropical forest (mean `199.07` trees/image; `24.7×` denser than internal test).

---

## 📁 Repository Structure

```text
TreeCountingUsingDroneImages/
├── assets/
│
├── datasets/
│   └── Tree-Top-View/                  # Primary Roboflow dataset
│       ├── train/                      # 1,191 train images & labels
│       ├── valid/                      # 149 valid images & labels
│       ├── test/                       # 55 test images & labels
│       ├── data.yaml
│       └── train_subset_595.txt        # 595 subset filelist
│
├── docs/                               # Presentations and project documentation
│   ├── Tree_Counting_Project_Review.pdf
│   ├── Tree_Counting_Project_Review.pptx
│   ├── yolov11_phase1_initialization.md
│   ├── yolov11_phase2_training.md
│   └── yolov8n_vs_yolo11n_comparison.md
│
├── External validation v5 and v7/      # YOLOv5n, YOLOv7-tiny & external validation suite
│   ├── external_eval/                  # Multi-model external generalization pipeline
│   │   ├── run_external_evaluation.py
│   │   ├── external_evaluation_summary.json
│   │   ├── per_image_external_counts_416.csv
│   │   ├── selvabox_test100.yaml
│   │   └── selvabox_test100_yolov7.yaml
│   ├── testdataset100/                 # External 100-tile SelvaBox benchmark metadata
│   │   ├── dataset_info.json
│   │   └── manifest.csv
│   ├── yolov5/                         # Ultralytics YOLOv5n (C3 anchor-free)
│   │   ├── train_yolov5.py
│   │   ├── evaluate_yolov5.py
│   │   ├── predict_yolov5.py
│   │   └── config_yolov5.yaml
│   ├── yolov7_tiny/                    # WongKinYiu YOLOv7-tiny (E-ELAN anchor-based)
│   │   ├── setup_yolov7_tiny.py
│   │   ├── patch_and_test_yolov7.py
│   │   ├── apply_train_patches.py
│   │   ├── train_yolov7_tiny.py
│   │   ├── evaluate_yolov7_tiny.py
│   │   ├── predict_yolov7_tiny.py
│   │   ├── config_yolov7_tiny.yaml
│   │   └── shims/                      # Python 3.13 lightweight compatibility shims
│   ├── prepare_subset.py               # Deterministic 50% stratified subset sampler
│   ├── prepare_testdataset100.py       # Asynchronous SelvaBox streaming curator
│   ├── monitor_both.py                 # Concurrent CPU training monitor
│   ├── subset_595_manifest.txt         # Exact image manifest for fair comparison
│   └── data_subset_595.yaml            # YAML dataset definition for v5 & v7-tiny
│
├── models/                             # Trained model checkpoints
│
├── notebooks/
│
├── outputs/
│   ├── metrics/                        # Confusion matrices, PR curves, counting CSVs
│   ├── predictions/                    # Qualitative model inference overlays
│   └── visualizations/
│
├── src/                                # Core YOLOv8 & YOLOv11 pipeline modules
│   ├── initialize_yolov11.py           # YOLOv11 Phase 1 initialization & verification
│   ├── train_yolov11.py                # YOLOv11 Phase 2 training pipeline
│   ├── evaluate_yolov11.py             # YOLOv11 test split evaluation
│   ├── predict_yolov11.py              # YOLOv11 inference and tree counting
│   ├── analyze_predictions_yolov11.py  # IoU error analysis (TP / FP / FN overlays)
│   ├── evaluate_counting.py            # Per-image counting MAE / RMSE comparison
│   ├── evaluate.py                     # Baseline YOLOv8n test evaluation
│   ├── predict.py                      # Baseline YOLOv8n prediction script
│   └── analyze_predictions.py          # Baseline YOLOv8n error analysis
│
├── train.py                            # Root training entry point
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 📊 Experimental Results & Benchmarks

### 1. In-Domain Validation Split Performance (`149` Images, `1,318` GT Trees)

| Model | Architecture Type | Training Split (`Images / Trees`) | Epochs | Precision | Recall | **mAP@0.50** | **mAP@0.50:0.95** | Training Wall Time (CPU) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n (Phase 2)** | Modern C3k2 Anchor-Free | `1,191 / 10,002` (100%) | 35 | **0.6327** | **0.6395** | **0.6479** | **0.3490** | 3.01 hrs |
| **YOLO11n (Phase 1)** | Modern C3k2 Anchor-Free | `1,191 / 10,002` (100%) | 30 | 0.6305 | 0.6366 | 0.6462 | 0.3407 | 1.24 hrs |
| **YOLOv8n (Baseline)** | C2f Anchor-Free | `1,191 / 10,002` (100%) | 30 | 0.6256 | 0.6252 | 0.6401 | 0.3358 | 1.76 hrs |
| **YOLOv5n (`yolov5nu`)** | C3 Anchor-Free | `595 / 5,127` (50%) | 30 | 0.6263 | 0.5877 | 0.6262 | 0.3204 | **33.0 mins** |
| **YOLOv7-tiny** | E-ELAN Anchor-Based | `595 / 5,127` (50%) | 30 | 0.5809 | 0.6199 | 0.5944 | 0.2784 | 70.5 mins |

---

### 2. Held-Out Internal Test Split Evaluation (`55` Images, `443` GT Trees, `imgsz=416`)

Models evaluated using official validators (`batch=8`, PR-curve integration) and operational counting pipelines (`conf=0.25, IoU=0.45`):

| Model | Training Split | Official Test Precision | Official Test Recall | **Official Test mAP@0.50** | **Official Test mAP@0.50:0.95** | CPU Latency (`batch=8`) | Pred Trees (`conf=0.25`) | Net Counting Bias | **Counting MAE** | Counting RMSE | `F1@0.50` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n (Phase 2)** | `1,191` (100%) | **0.5764** | 0.6082 | **0.6083** | **0.2993** | 48.02 ms | 444 | **+0.23%** (Near Zero) | 3.16 | 4.10 | 0.5953 |
| **YOLOv8n (Baseline)** | `1,191` (100%) | 0.5288 | **0.6206** | 0.6000 | 0.2914 | 34.74 ms | 460 | +3.84% | **3.04** | **4.06** | 0.5891 |
| **YOLOv5n (`yolov5nu`)** | `595` (50%) | 0.5566 | 0.5801 | 0.5872 | 0.2683 | **27.14 ms** | 449 | +1.35% | 3.42 | 4.38 | **0.5964** |
| **YOLOv7-tiny** | `595` (50%) | 0.5142 | 0.5959 | 0.5450 | 0.2463 | 72.05 ms | 692 | +56.21% | 5.42 | 6.87 | 0.5234 |

#### Key In-Domain Insights:
- **YOLO11n Phase 2** achieves the highest detection accuracy (**`mAP@0.50 = 0.6083`**, **`mAP@0.50:0.95 = 0.2993`**) and virtually eliminates systematic counting bias (**+0.23% net bias**, predicting 444 trees vs. 443 ground truth).
- **YOLOv5n** trained on only 50% data shows exceptional compute efficiency: achieves 0.5872 mAP@0.50 in only 33 minutes of CPU training, with the lowest CPU inference latency (**27.14 ms**).
- **YOLOv7-tiny** exhibits higher recall under lower confidence but suffers from anchor duplicate predictions on internal data, resulting in +56.2% over-counting at default thresholds.

For detailed YOLOv8n vs. YOLO11n logs and comparison reports:
- 📄 **Full Comparison Report:** [docs/yolov8n_vs_yolo11n_comparison.md](docs/yolov8n_vs_yolo11n_comparison.md)
- 📄 **Training & Validation Log:** [docs/yolov11_phase2_training.md](docs/yolov11_phase2_training.md)

---

### 3. External Zero-Shot Generalization (`CanopyRS / SelvaBox`, `100` Tiles, `8,111` GT Trees)

Cross-domain generalization test on unseen neotropical rainforest orthomosaics across two inference resolutions (`416×416` vs. `640×640`):

| Model | Eval `imgsz` | Validator Precision | Validator Recall | **Validator mAP@0.50** | **Validator mAP@0.50:0.95** | Single-Stream Latency (`batch=1`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLOv7-tiny** | 416 | 0.4237 | **0.2800** | **0.2642** | **0.0963** | 138.49 ms |
| **YOLOv8n** | 416 | 0.5049 | 0.1626 | 0.1835 | 0.0692 | 94.44 ms |
| **YOLO11n** | 416 | 0.4712 | 0.1769 | 0.1786 | 0.0691 | 89.82 ms |
| **YOLOv5n** | 416 | **0.5081** | 0.1632 | 0.1767 | 0.0679 | **85.59 ms** |
| --- | --- | --- | --- | --- | --- | --- |
| **YOLOv7-tiny** | 640 | 0.4495 | **0.3321** | **0.3058** (+15.7%) | **0.1177** (+22.2%) | 264.73 ms |
| **YOLOv5n** | 640 | 0.4868 | 0.2396 | 0.2303 (+30.3%) | 0.0921 (+35.6%) | **111.81 ms** |
| **YOLOv8n** | 640 | 0.4515 | 0.2335 | 0.2295 (+25.1%) | 0.0943 (+36.3%) | 127.35 ms |
| **YOLO11n** | 640 | **0.5172** | 0.2047 | 0.2258 (+26.4%) | 0.0925 (+33.9%) | 123.19 ms |

#### Ecological Strata Breakdown:
- **Ecuador (Tiputini - Moderate Canopy Density, mean `18.25` trees/img):**
  - **`YOLOv7-tiny` (`@640, conf=0.15`):** Achieves **`MAE = 3.42` trees/image**, Pearson **`r = 0.9391`**, and **`+4.84%` net bias**.
  - **`YOLOv5n` (`@640, conf=0.10`):** Achieves **`MAE = 4.15` trees/image**, Pearson **`r = 0.9002`**.
  - **`YOLOv8n` (`@640, conf=0.10`):** Achieves **`MAE = 4.68` trees/image**, Pearson **`r = 0.8688`**.
  - **`YOLO11n` (`@640, conf=0.10`):** Achieves **`MAE = 5.23` trees/image**, Pearson **`r = 0.8884`**.
- **Brazil (ZF2 - Closed Canopy, mean `126.23` trees/img):**
  - Ultralytics models maintain high precision (**80%–83%**) on dominant emergent crowns; `YOLOv7-tiny` achieves lowest counting error (**`MAE = 39.08`** at `@640, conf=0.10`).
- **Panama (Agua Salud - Hyper-Dense Canopy, mean `199.07` trees/img):**
  - Modern models detect top-layer crowns with high precision (**`85.71%` for YOLO11n**, **`81.21%` for YOLOv5n**). Multi-scale anchor boxes in `YOLOv7-tiny` capture smaller overlapping crowns, localizing `1,192` true positive trees.

---

## 🚀 Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/Nitheesh-Git/TreeCountingUsingDroneImages.git
cd TreeCountingUsingDroneImages

# Create and activate virtual environment
python -m venv .venv

# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Core Pipeline: Training & Evaluating YOLO11n (`src/`)

```bash
# Train YOLO11n on full Tree-Top-View dataset
python src/train_yolov11.py

# Evaluate YOLO11n on held-out test set
python src/evaluate_yolov11.py

# Run YOLO11n prediction & tree counting
python src/predict_yolov11.py

# Compare per-image counting accuracy (YOLOv8n vs YOLO11n)
python src/evaluate_counting.py
```

---

## 🔬 External Validation & Lightweight Models (`External validation v5 and v7/`)

### 1. Generating Fair CPU 50% Subset
```bash
python "External validation v5 and v7/prepare_subset.py"
```

### 2. Training YOLOv5n & YOLOv7-tiny

**YOLOv5n:**
```bash
python "External validation v5 and v7/yolov5/train_yolov5.py"
```

**YOLOv7-tiny (with Python 3.13 shims):**
```bash
# Setup & clone WongKinYiu repository
python "External validation v5 and v7/yolov7_tiny/setup_yolov7_tiny.py"

# Apply training compatibility patches
python "External validation v5 and v7/yolov7_tiny/apply_train_patches.py"

# Launch training
python "External validation v5 and v7/yolov7_tiny/train_yolov7_tiny.py"
```

### 3. Running External SelvaBox Generalization Benchmark
```bash
# Stream and curate 100-tile benchmark from Hugging Face
python "External validation v5 and v7/prepare_testdataset100.py"

# Run comprehensive multi-model benchmark at 416 & 640 resolutions
python "External validation v5 and v7/external_eval/run_external_evaluation.py"
```

---

## 📅 Project Milestone Status

| Phase | Milestone / Task | Status |
| :--- | :--- | :---: |
| **Phase 1** | Repository setup, environment configuration & data pipeline | ✅ Completed |
| **Phase 2** | Roboflow `Tree-Top-View` baseline model training (**YOLOv8n**) | ✅ Completed |
| **Phase 3** | SOTA architecture upgrade to **YOLO11n** (Phase 1 & Phase 2 in `src/`) | ✅ Completed |
| **Phase 4** | In-domain evaluation, IoU localization analysis & metric validation | ✅ Completed |
| **Phase 5** | Reproducible fair CPU subset preparation (`595` images) | ✅ Completed |
| **Phase 6** | Comparison benchmarking: **YOLOv5n** vs. **YOLOv7-tiny** | ✅ Completed |
| **Phase 7** | External zero-shot generalization benchmark (**CanopyRS / SelvaBox**) | ✅ Completed |
| **Phase 8** | Final comprehensive project documentation & reporting | ✅ Completed |

---

## 👥 Team & Contributions

| Member | Primary Responsibilities |
| :--- | :--- |
| **Nitheesh** | Lead Model Development, Architecture Training Pipeline, Repository Management |
| **Nishu Kumari** | Literature Survey, Dataset Analysis, Ecological Evaluation & Documentation |
| **Astley** | Data Engineering, Model Benchmarking, Generalization Testing & Verification |

---

## 📚 References & Acknowledgments

- **Dataset (In-Domain):** Roboflow Universe – [Tree-Top-View Dataset](https://universe.roboflow.com/) (CC BY 4.0).
- **Dataset (External):** Baudchon et al., *CanopyRS / SelvaBox: Neotropical Tree Crown Benchmark*, Hugging Face `CanopyRS/SelvaBox`.
- **YOLO Implementations:**
  - Ultralytics YOLOv8 & YOLO11: [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)
  - WongKinYiu YOLOv7: [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
  - Ultralytics YOLOv5: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)

---

## 📄 License

This project uses the **Tree-Top-View Dataset** under the **CC BY 4.0 License**. Please refer to individual dataset repositories for further attribution and licensing requirements.
