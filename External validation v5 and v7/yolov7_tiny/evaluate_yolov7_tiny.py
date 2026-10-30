import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
YOLOV7_REPO = SCRIPT_DIR / "yolov7_repo"
SHIMS_DIR = SCRIPT_DIR / "shims"

sys.path.insert(0, str(SHIMS_DIR))
sys.path.insert(0, str(YOLOV7_REPO))

import cv2
import numpy as np
import torch
import yaml
from models.experimental import attempt_load
from utils.datasets import letterbox, create_dataloader
from utils.general import non_max_suppression, scale_coords, xywh2xyxy, box_iou, check_img_size
from utils.metrics import ap_per_class
from utils.torch_utils import select_device, time_synchronized
import test as yolov7_test

def parse_args():
    default_weights = str(SCRIPT_DIR / "runs" / "yolov7_tiny_subset" / "weights" / "best.pt")
    default_data = str(SCRIPT_DIR / "data_subset_595_yolov7.yaml")
    default_project = str(SCRIPT_DIR / "runs" / "evaluation")

    parser = argparse.ArgumentParser(description="Evaluate trained YOLOv7-tiny on validation or test dataset.")
    parser.add_argument("--weights", type=str, default=default_weights, help="Path to trained weights")
    parser.add_argument("--data", type=str, default=default_data, help="Path to dataset yaml")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--project", type=str, default=default_project, help="Evaluation output project")
    parser.add_argument("--name", type=str, default="yolov7_tiny_test_eval", help="Evaluation run name")
    parser.add_argument("--imgsz", type=int, default=416, help="Image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device")
    parser.add_argument("--eval-counting", action="store_true", default=True, help="Compute Tree Counting MAE")
    return parser.parse_args()

def evaluate_counting_mae(model, images_dir, labels_dir, imgsz=416, conf=0.25, iou_thres=0.45, device="cpu"):
    img_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not img_paths:
        return None, None, None

    total_gt = 0
    total_pred = 0
    abs_errors = []

    for p in img_paths:
        basename = os.path.basename(p)
        lbl_path = os.path.join(labels_dir, Path(basename).stem + ".txt")

        gt_count = 0
        if os.path.exists(lbl_path):
            with open(lbl_path, "r", encoding="utf-8") as f:
                gt_count = sum(1 for line in f if line.strip())

        img0 = cv2.imread(p)
        if img0 is None:
            continue
        img = letterbox(img0, imgsz, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img_tensor = torch.from_numpy(img).to(device).float() / 255.0
        if img_tensor.ndimension() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = non_max_suppression(pred, conf_thres=conf, iou_thres=iou_thres)[0]

        pred_count = len(pred) if pred is not None else 0
        total_gt += gt_count
        total_pred += pred_count
        abs_errors.append(abs(pred_count - gt_count))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
    return mae, total_gt, total_pred

def main():
    args = parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🌲 YOLOv7-tiny Model Evaluation & Metrics Report 🌲")
    print("=" * 70)
    print(f"Weights path     : {args.weights}")
    print(f"Dataset config   : {args.data}")
    print(f"Evaluation split : {args.split}")
    print(f"Image size       : {args.imgsz}")
    print(f"Device           : {args.device}")
    print(f"Output directory : {save_dir}")
    print("=" * 70)

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    device = select_device(args.device)
    model = attempt_load(str(weights_path), map_location=device).float()
    model.eval()

    with open(args.data) as f:
        data_dict = yaml.safe_load(f)

    # Set opt for test.py
    opt = argparse.Namespace()
    opt.data = args.data
    opt.weights = str(weights_path)
    opt.batch_size = args.batch
    opt.img_size = [args.imgsz, args.imgsz]
    opt.conf_thres = args.conf_thres
    opt.iou_thres = args.iou_thres
    opt.save_json = False
    opt.single_cls = False
    opt.augment = False
    opt.verbose = True
    opt.model = model
    opt.dataloader = None
    opt.save_dir = save_dir
    opt.save_txt = False
    opt.save_hybrid = False
    opt.save_conf = False
    opt.plots = True
    opt.wandb_logger = None
    opt.compute_loss = None
    opt.half_precision = False
    opt.trace = False
    opt.is_coco = False
    opt.v5_metric = True
    opt.task = "test" if args.split == "test" else "val"
    opt.device = args.device
    opt.project = args.project
    opt.name = args.name
    opt.exist_ok = True

    gs = max(int(model.stride.max()), 32)
    imgsz = check_img_size(args.imgsz, s=gs)
    dataloader = create_dataloader(
        data_dict[opt.task], imgsz, args.batch, gs, opt, pad=0.5, rect=True, prefix=f"{opt.task}: "
    )[0]

    t_start = time.time()
    results, _, times = yolov7_test.test(
        data=data_dict,
        batch_size=args.batch,
        imgsz=imgsz,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        model=model,
        dataloader=dataloader,
        single_cls=False,
        save_dir=save_dir,
        plots=False,
        v5_metric=True
    )
    total_eval_time = time.time() - t_start

    mp, mr, map50, map50_95 = results[0], results[1], results[2], results[3]
    inf_latency = float(times[0]) if len(times) > 0 else 0.0
    nms_latency = float(times[1]) if len(times) > 1 else 0.0
    total_latency = float(times[2]) if len(times) > 2 else inf_latency + nms_latency

    print("\n" + "=" * 70)
    print(f"📊 YOLOv7-tiny EVALUATION RESULTS ({args.split.upper()} SPLIT) 📊")
    print("=" * 70)
    print(f"Precision (P)    : {mp:.4f}")
    print(f"Recall (R)       : {mr:.4f}")
    print(f"mAP@0.50         : {map50:.4f}")
    print(f"mAP@0.50:0.95    : {map50_95:.4f}")
    print(f"Inference Latency: {inf_latency:.2f} ms/image")
    print(f"NMS Postprocess  : {nms_latency:.2f} ms/image")
    print(f"Total Latency    : {total_latency:.2f} ms/image")
    print("=" * 70)

    counting_mae = None
    if args.eval_counting:
        split_name = "test" if args.split == "test" else "valid"
        images_dir = REPO_ROOT / "datasets" / "Tree-Top-View" / split_name / "images"
        labels_dir = REPO_ROOT / "datasets" / "Tree-Top-View" / split_name / "labels"

        if images_dir.exists() and labels_dir.exists():
            print(f"\n[INFO] Evaluating tree-counting MAE on {split_name} split...")
            counting_mae, total_gt, total_pred = evaluate_counting_mae(
                model, str(images_dir), str(labels_dir), imgsz=args.imgsz, conf=0.25, iou_thres=0.45, device=device
            )
            print(f"Total Ground-Truth Trees : {total_gt}")
            print(f"Total Predicted Trees    : {total_pred}")
            print(f"Tree Counting MAE        : {counting_mae:.3f} trees/image")
            print("=" * 70)

    summary = {
        "model": "YOLOv7-tiny",
        "weights": str(weights_path),
        "split": args.split,
        "precision": round(float(mp), 4),
        "recall": round(float(mr), 4),
        "map50": round(float(map50), 4),
        "map50_95": round(float(map50_95), 4),
        "inference_ms": round(float(inf_latency), 2),
        "postprocess_ms": round(float(nms_latency), 2),
        "total_ms": round(float(total_latency), 2),
        "counting_mae": round(float(counting_mae), 3) if counting_mae is not None else None,
        "total_gt_trees": int(total_gt) if counting_mae is not None else None,
        "total_pred_trees": int(total_pred) if counting_mae is not None else None
    }

    summary_file = save_dir / "metrics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Summary written to: {summary_file}")

if __name__ == "__main__":
    main()
