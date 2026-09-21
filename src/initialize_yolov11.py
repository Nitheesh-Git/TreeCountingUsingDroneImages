import os
from ultralytics import YOLO
import sys

def main():
    print("=" * 60)
    print("YOLOv11 Phase 1 Initialization & Smoke Test")
    print("=" * 60)

    model_name = "yolo11n.pt"
    test_image_path = "datasets/Tree-Top-View/test/images/2018_SJER_3_252000_4106000_image_234_jpeg.rf.e24a2287aeb84398f98161045cece665.jpg"
    output_dir = "outputs/yolov11_initialization"
    conf_threshold = 0.25

    # 1. Load the model
    try:
        print(f"[INFO] Loading YOLOv11 model: {model_name}...")
        model = YOLO(model_name)
        print("[SUCCESS] YOLOv11 model loaded successfully.")
        
        # Print basic model info securely (avoiding massive dumps)
        print(f"[INFO] Model task: {model.task}")
        print(f"[INFO] Model names: {model.names}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load YOLOv11 model: {e}")
        sys.exit(1)

    # 2. Verify test image exists
    if not os.path.exists(test_image_path):
        print(f"[ERROR] Test image not found at: {test_image_path}")
        sys.exit(1)

    # 3. Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # 4. Run Smoke Test Inference
    try:
        print(f"[INFO] Running inference smoke test on: {test_image_path}")
        print(f"[INFO] Confidence threshold set to: {conf_threshold}")
        
        # We run inference. `project` and `name` route the saved output properly without touching YOLOv8 folders.
        results = model.predict(
            source=test_image_path,
            conf=conf_threshold,
            save=True,
            project=output_dir,
            name="smoke_test",
            exist_ok=True
        )
        
        if len(results) > 0:
            num_detections = len(results[0].boxes)
            print(f"[SUCCESS] Inference complete. Detected {num_detections} objects.")
            print(f"[INFO] Annotated image saved in: {os.path.join(output_dir, 'smoke_test')}")
        else:
            print("[WARNING] Inference completed but returned no result objects.")
            
    except Exception as e:
        print(f"[ERROR] Inference smoke test failed: {e}")
        sys.exit(1)

    print("=" * 60)
    print("Phase 1 Initialization successfully completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
