# YOLOv11 Upgrade - Phase 1: Initialization

## 1. Objective
The objective of Phase 1 is to securely prepare the environment, verify YOLO11 support within the existing Ultralytics installation, and successfully initialize the YOLOv11 model architecture. This phase ensures the new model can load and run a basic smoke test on the existing dataset without breaking the existing YOLOv8n baseline or inadvertently modifying previous results.

## 2. YOLOv11 Model Variant
- **Model:** YOLOv11 Nano (`yolo11n.pt`)
- **Reason:** Selected to maintain a fair performance, inference speed, and footprint comparison against the previous `yolov8n.pt` baseline in our CPU-constrained environment.

## 3. Environment Details
- **Operating System:** Windows 11 Home
- **CPU:** 12th Gen Intel(R) Core(TM) i7-1255U
- **Device:** CPU (No CUDA/NVIDIA GPU available)

## 4. Dependencies
- **Python:** 3.12+ (managed via `.venv`)
- **PyTorch:** 2.14.0+cpu
- **Ultralytics:** 8.4.149 (Verified to support YOLOv11)

## 5. Initialization Process
A custom Python script was created to test the initialization. The script programmatically loaded the `yolo11n.pt` weights, outputted model metadata securely, and executed a single inference smoke test against a known image from the test dataset. The script was explicitly designed *not* to invoke the `.train()` routine. 

## 6. Script Location
`src/initialize_yolov11.py`

## 7. Execution Command
```bash
python src/initialize_yolov11.py
```

## 8. Inference Smoke-Test Command
The smoke test was embedded directly inside the initialization script via the `model.predict(...)` method using a fixed confidence threshold of `0.25`.

## 9. Verification Results
- **Model Load:** Successful.
- **Model Task:** `detect`
- **Smoke Test Image:** `datasets/Tree-Top-View/test/images/2018_SJER_3_252000_4106000_image_234_jpeg.rf.e24a2287aeb84398f98161045cece665.jpg`
- **Execution:** Successful (script exited with code 0 without exceptions).

## 10. Output Locations
- **Smoke Test Annotated Image:** `outputs/yolov11_initialization/smoke_test/`
- *(Note: The downloaded `yolo11n.pt` weights were deliberately kept out of Git tracking to prevent repository bloat).*

## 11. What Was NOT Performed
- **Training:** YOLOv11 training was NOT started.
- **Hyperparameter Tuning:** No tuning has been performed.
- **Baseline Disruption:** The YOLOv8n baseline results, scripts, and model weights were explicitly NOT changed or modified.
- **Model Weights Commit:** Model weights were NOT committed to the Git repository.

## 12. Known Limitations
- The model currently being initialized is pre-trained on the COCO dataset (80 generic object classes). It has not yet been fine-tuned to recognize the specific `tree-top` class in our drone dataset. Any detections in the smoke test are purely generic artifacts until Phase 2 is executed.

## 13. Next Planned Phase
**Phase 2: YOLOv11 Training and Hyperparameter Tuning.** 
Once this Phase 1 initialization PR is reviewed and merged, Phase 2 will execute the `.train()` loop using the dataset configuration, comparing metrics directly against the YOLOv8n baseline.
