# 🌳 Tree Counting Using Drone Images

An AI-powered computer vision project for detecting and counting trees from aerial drone imagery using **Ultralytics YOLOv11**.

---

# 📌 Project Overview

Manual tree counting is time-consuming, labor-intensive, and prone to human error. This project automates the process using deep learning-based object detection on top-view drone images.

The system is designed to detect individual tree canopies, estimate the total number of trees, and provide a scalable solution for forestry, plantation management, agriculture, and environmental monitoring.

---

# 🎯 Objectives

- Automatically detect trees from drone images.
- Count the total number of trees.
- Train a robust YOLOv11 object detection model.
- Evaluate model performance using standard object detection metrics.
- Build a complete and reproducible deep learning pipeline.

---

# 🛠 Technology Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python 3.x |
| Deep Learning | PyTorch |
| Object Detection | Ultralytics YOLOv11 |
| Image Processing | OpenCV |
| Data Analysis | NumPy, Pandas |
| Visualization | Matplotlib |
| Development | VS Code, Jupyter Notebook |
| Version Control | Git & GitHub |

---

# 📂 Dataset

**Dataset Name**

Tree-Top-View Dataset

**Source**

Roboflow Universe

**Task**

Object Detection

**Dataset Format**

YOLOv11

**Images**

1362 Images

**Classes**

- tree-top

**License**

CC BY 4.0

Dataset Location

```text
datasets/Tree-Top-View/
```

---

# 📁 Repository Structure

```text
TreeCountingUsingDroneImages/
│
├── assets/
│
├── datasets/
│   └── Tree-Top-View/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── valid/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       │   ├── images/
│       │   └── labels/
│       ├── data.yaml
│       ├── README.dataset.txt
│       └── README.roboflow.txt
│
├── docs/
│
├── models/
│   └── checkpoints/
│
├── notebooks/
│
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   └── visualizations/
│
├── scripts/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🔄 Project Workflow

```text
Drone Images
      │
      ▼
Dataset Preparation
      │
      ▼
Data Exploration
      │
      ▼
Model Training (YOLOv11)
      │
      ▼
Prediction
      │
      ▼
Tree Detection
      │
      ▼
Tree Counting
      │
      ▼
Evaluation
      │
      ▼
Visualization
```

---

# 🚀 Getting Started

## Clone Repository

```bash
git clone git@github.com:Nitheesh-Git/TreeCountingUsingDroneImages.git
cd TreeCountingUsingDroneImages
```

## Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🏋️ Training

```bash
yolo detect train \
model=yolo11n.pt \
data=datasets/Tree-Top-View/data.yaml \
epochs=100 \
imgsz=640
```

---

# 📈 Evaluation Metrics

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95
- Tree Counting Accuracy
- Inference Speed

---

# 📤 Expected Output

- Automatic tree detection
- Automatic tree counting
- Bounding box visualization
- Annotated prediction images
- Performance metrics
- Trained YOLOv11 model

---

# 📅 Project Roadmap

| Phase | Status |
|--------|--------|
| Repository Setup | ✅ |
| Dataset Collection | ✅ |
| Literature Survey | ⏳ |
| Dataset Analysis | ⏳ |
| Model Training | ⏳ |
| Hyperparameter Tuning | ⏳ |
| Evaluation | ⏳ |
| Documentation | ⏳ |
| Final Presentation | ⏳ |

---

# 👥 Team

| Member | Responsibility |
|----------|----------------|
| Nitheesh | Project Lead, Model Development, Documentation |
| Team Members | Dataset Analysis, Training, Evaluation, Documentation |

---

# 📚 References

- Roboflow Universe - Tree-Top-View Dataset
- Ultralytics YOLO Documentation
- PyTorch Documentation

---

# 📄 License

This project uses the **Tree-Top-View Dataset** under the **CC BY 4.0 License**. Please refer to the dataset documentation for attribution requirements.
