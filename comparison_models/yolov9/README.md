# YOLOv9t

Comparison model for the Tree Counting Using Drone Images project. The shared protocol
is in [../README.md](../README.md). The model is trained on the **594-image half
subset**, while the YOLOv8n and YOLO11n baselines used the full 1,191-image split.

- **Variant:** YOLOv9t (`yolov9t.pt`), COCO-pretrained, official Ultralytics asset
- **Implementation:** `ultralytics==8.4.177` (upstream architecture: WongKinYiu/yolov9)
- **Architecture note:** GELAN backbone (YOLOv9 family). The Ultralytics port is trained with the standard Ultralytics detection loss, not the upstream repo's dual-branch PGI training script (to be confirmed against the installed model yaml).
- **Settings:** `config.yaml` (416 px, 30 epochs, patience 5, batch 8, seed 0, optimizer auto)

## Commands

```bash
python comparison_models/yolov9/train.py --time 0.5
python comparison_models/yolov9/evaluate.py --split val
python comparison_models/yolov9/evaluate.py --split test
python comparison_models/yolov9/predict.py datasets/Tree-Top-View/test/images/<image>.jpg
```

## Outputs

- `metrics/train_summary.json`: epochs completed, best epoch, duration, environment
- `metrics/{val,test}_metrics.json`: precision, recall, mAP50, mAP50-95, counting MAE, CPU latency
- `metrics/{val,test}_counting.csv`: per-image ground-truth vs predicted counts
- `runs/` (git-ignored): checkpoints, training curves, confusion matrices

## Results

Training: 17 epochs in 30.6 min on CPU (stopped by the 0.5 h time budget, run at the same time as the other comparison model). Best validation epoch: 16.

| Split | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Predicted / GT trees |
|---|---|---|---|---|---|---|
| Validation (116) | 0.5294 | 0.4839 | 0.5188 | 0.2719 | 3.3966 | 850 / 932 |
| Test (55) | 0.5052 | 0.5532 | 0.5256 | 0.2640 | 2.8545 | 428 / 443 |

CPU inference latency (12th Gen Intel(R) Core(TM) i5-12500H, batch 1, 416 px): 21.5 ± 0.45 ms/img (total 22.2 ± 0.47 ms).

The cross-model comparison is in [../RESULTS.md](../RESULTS.md).
