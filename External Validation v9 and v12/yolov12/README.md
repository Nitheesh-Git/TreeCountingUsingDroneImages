# YOLOv12n

Comparison model for the Tree Counting Using Drone Images project. The shared protocol
is in [../README.md](../README.md). The model is trained on the **594-image half
subset** of the 1,191-image training split, under a 0.5 h CPU time budget (17 epochs).

- **Variant:** YOLOv12n (`yolo12n.pt`), COCO-pretrained, official Ultralytics asset
- **Implementation:** `ultralytics==8.4.177` (upstream architecture: sunsmarterjie/yolov12)
- **Architecture note:** Attention-centric design (area attention, R-ELAN blocks). Ultralytics uses FlashAttention when installed and otherwise falls back to PyTorch scaled-dot-product attention; this setup uses the fallback (no flash-attn install).
- **Settings:** `config.yaml` (416 px, 30 epochs, patience 5, batch 8, seed 0, optimizer auto)

## Commands

```bash
python "External Validation v9 and v12/yolov12/train.py" --time 0.5
python "External Validation v9 and v12/yolov12/evaluate.py" --split val
python "External Validation v9 and v12/yolov12/evaluate.py" --split test
python "External Validation v9 and v12/yolov12/predict.py" datasets/Tree-Top-View/test/images/<image>.jpg
```

## Outputs

- `metrics/train_summary.json`: epochs completed, best epoch, duration, environment
- `metrics/{val,test}_metrics.json`: precision, recall, mAP50, mAP50-95, counting MAE, CPU latency
- `metrics/{val,test}_counting.csv`: per-image ground-truth vs predicted counts
- `runs/` (git-ignored): checkpoints, training curves, confusion matrices

## Results

Training: 17 epochs in 30.0 min on CPU (stopped by the 0.5 h time budget, run at the same time as the other comparison model). Best validation epoch (saved as `best.pt`, Ultralytics fitness = mAP50-95): 16.

| Split | Precision | Recall | mAP50 | mAP50-95 | Count MAE | Predicted / GT trees |
|---|---|---|---|---|---|---|
| Validation (116) | 0.5309 | 0.5440 | 0.5525 | 0.2761 | 2.9914 | 855 / 932 |
| Test (55) | 0.5219 | 0.5421 | 0.5383 | 0.2596 | 3.0364 | 394 / 443 |

CPU inference latency (12th Gen Intel(R) Core(TM) i5-12500H, batch 1, 416 px): 21.19 ± 0.58 ms/img (total 21.88 ± 0.58 ms).

The cross-model comparison is in [../RESULTS.md](../RESULTS.md).
