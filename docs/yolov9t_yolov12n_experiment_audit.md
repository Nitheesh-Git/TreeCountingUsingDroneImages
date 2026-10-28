# Technical Audit: YOLOv9t and YOLOv12n Comparison Experiments

**Project:** Tree Counting Using Drone Images (`TreeCountingUsingDroneImages`)
**Machine:** Linux laptop, `/home/nitheesh/tree-counting-using-drone-images`
**Branch:** `yolov12-yolov9-comparison` (base `main` @ `356c0a1`, first comparison commit `62b4dce`)
**Audit date:** 29 October 2026
**Companion documents:** `comparison_models/REPORT.md` (results report), `comparison_models/RESULTS.md` (short table),
and Astley's `docs/comprehensive_model_experiment_audit.md` (YOLOv5n, YOLOv7-tiny).

Every number below was read from files produced on this machine:
- `results.csv` and `args.yaml` in each `runs/` folder
- `train.log`
- `metrics/*.json` and `metrics/*.csv`
- the checkpoint files on disk

Numbers for the other four models are quoted from the companion documents and are marked as such.

---

## Executive Summary

1. **Scope.**
   - Two more comparison models were trained and evaluated: **YOLOv9t** (`yolov9t.pt`) and **YOLOv12n** (`yolo12n.pt`).
   - Both use the Ultralytics trainer (the same engine as YOLOv5n), on the Tree-Top-View dataset at 416 × 416.
   - Training data was a reproducible **594-image half subset** of the 1,191-image training split.
   - Evaluation used the **unchanged** 116-image validation split and 55-image test split.
2. **Deviation from the planned protocol.**
   - A deadline and a very slow network (the CPU PyTorch install alone took about 70 minutes) prevented the planned 30-epoch runs.
   - Each model was trained under a **0.5 h CPU time budget** (`--time 0.5`), and **both trained at the same time** with 8 threads each.
   - Ultralytics fitted the schedule to the budget, giving **17 epochs** for each model.
   - This deviation is stated in every results file.
3. **Held-out test results (55 images, 443 trees).**

   | Model | P | R | mAP@0.50 | mAP@0.50:0.95 | Count MAE | Predicted trees |
   |---|---|---|---|---|---|---|
   | YOLOv9t | 0.5052 | 0.5532 | 0.5256 | 0.2640 | **2.8545** | 428 (−15, −3.39%) |
   | YOLOv12n | 0.5219 | 0.5421 | 0.5383 | 0.2596 | 3.0364 | 394 (−49, −11.06%) |

4. **Comparison between the two models, and with the other half-subset models.**
   - **Detection:** YOLOv12n is the better detector of the pair. It is ahead on every validation metric (mAP@0.50 0.5525 vs 0.5188), and on test precision, F1 and mAP@0.50.
   - **Counting:** YOLOv9t has the lower test counting MAE (2.85 vs 3.04). That is also the lowest among the four half-subset models (YOLOv5n 3.20, YOLOv7-tiny 3.76), despite the least training: 17 epochs, 0.5 h.
   - **False positives:** YOLOv12n has 32 fewer false positives than YOLOv9t on the test split (170 vs 202) while finding almost the same number of trees.
5. **Subset difference with YOLOv5n and YOLOv7-tiny.** YOLOv9t and YOLOv12n did **not** use the same 595-image subset as YOLOv5n and YOLOv7-tiny (see Part 4.4):
   - Their subset was drawn on Astley's machine with `seed=42`, sampling individual files.
   - The subset here was built independently with `seed=0`, sampling Roboflow augmentation groups, before that subset was visible to this machine.
   - Comparisons across these two groups therefore also include a difference in which training images were used.
6. **External dataset (SelvaBox `testdataset100`, zero-shot).** Both models were evaluated at a pre-specified 832 px (ground-resolution matched) and at 416 px (Part 16).
   - At 832 px: YOLOv9t mAP@0.50 0.2593, YOLOv12n 0.2399.
   - Both under-count heavily: 1,450 and 1,748 of 8,111 trees predicted, count MAE 66.8 and 63.9.
   - The cause is dense, touching tropical crowns, not scale: crown sizes in metres match the training data.

---

## Part 1 — Hardware and software environment

| Component | Value | Notes |
|---|---|---|
| OS | Linux (kernel 7.0.0-38-generic) | |
| CPU | 12th Gen Intel Core i5-12500H, 16 logical CPUs | Astley's machine: i7-1255U, 12 logical |
| RAM | 15 GiB, about 8 GiB available during runs | |
| GPU | NVIDIA RTX 3050 Laptop, 4 GB, **not used** | The CUDA PyTorch download (about 4 GB) could not finish on the available network (0.05–1 MB/s) |
| Python | 3.12.13 (uv-managed `.venv`) | Astley: 3.13.5 |
| torch / torchvision | 2.14.1+cpu / 0.29.2+cpu | Astley: 2.14.0+cpu / 0.29.0+cpu |
| ultralytics | **8.4.177** (pinned, released 2026-10-13) | Astley: 8.4.149 |
| Training threads | `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8` per process, `workers=2` | Two processes running at the same time |

The project `.venv` is git-ignored. Nothing was installed into the system Python.

---

## Part 2 — Directory structure created

```text
comparison_models/
├── README.md                     protocol, settings, commands
├── RESULTS.md / results_summary.csv   short four-model table
├── REPORT.md                     detailed five-parameter report (for the final report)
├── common/
│   ├── make_subset.py            builds the 594-image subset (seed 0)
│   ├── subset_manifest.txt       the 594 selected images (repo-relative paths)
│   ├── subset_stats.md / .json   subset vs full-split statistics and label validation
│   ├── pipeline.py               shared train / evaluate / count / latency code
│   ├── cli.py                    shared command-line arguments
│   ├── report_details.py         TP/FP/FN, complexity, counting breakdown
│   ├── evaluate_external.py      zero-shot SelvaBox testdataset100 evaluation
│   └── summarize.py              writes RESULTS.md and results_summary.csv
├── yolov9/  yolov12/
│   ├── config.yaml               model + training settings
│   ├── train.py / evaluate.py / predict.py   thin wrappers around common/
│   ├── README.md                 per-model notes and results
│   ├── metrics/                  train_summary.json, {val,test}_metrics.json,
│   │                             {val,test}_counting.csv, report_details.json   (tracked)
│   ├── weights/                  pretrained checkpoint      (git-ignored)
│   ├── runs/                     training run, best.pt/last.pt, train.log   (git-ignored)
│   └── predictions/              annotated images           (git-ignored)
├── external_dataset/README.md    reserved for held-out external data
└── data/tree_top_view_half/      generated symlink dataset view   (git-ignored)
```

**Not modified:** `datasets/Tree-Top-View/` (including `data.yaml` and every image and label), `train.py`, every script in `src/`, `outputs/`, `docs/` (except for this new file), and the existing checkpoints. The only change to a pre-existing file is 8 lines appended to `.gitignore`.

---

## Part 3 — Model selection and implementation verification

| Model | Variant used | Implementation | Upstream architecture | Pretrained checkpoint | Weight transfer |
|---|---|---|---|---|---|
| YOLOv9t | `yolov9t` (tiny), no substitution | Ultralytics 8.4.177 `DetectionModel` | WongKinYiu/yolov9, GELAN | `yolov9t.pt`, Ultralytics assets v8.3.0, COCO, 4,969,803 B | 1,333 / 1,339 tensors (the 6 head tensors were reset for nc = 1) |
| YOLOv12n | `yolo12n` (nano), no substitution | Ultralytics 8.4.177 `DetectionModel` | sunsmarterjie/yolov12, area attention + R-ELAN | `yolo12n.pt`, Ultralytics assets v8.3.0, COCO, 5,595,063 B | 685 / 691 tensors |

**Why the Ultralytics implementations were used:**
- Both models are native in Ultralytics. They use the same data loader, augmentation pipeline, TAL assigner, CIoU + DFL + BCE loss, `optimizer=auto` logic and metric code as YOLOv5nu, so implementation differences stay as small as possible.
- No dependency patches were needed, unlike YOLOv7-tiny.
- The upstream repos ship their own trainers and newer checkpoints, such as the YOLOv12 "turbo" weights and YOLOv9's dual-branch PGI training. These were **not** used.
- YOLOv12n ran without FlashAttention (not installed). Ultralytics falls back to PyTorch scaled-dot-product attention.

**Model complexity** (from the Ultralytics model summary in `train.log` and `report_details.json`):

| Model | Layers (unfused / fused) | Parameters (unfused / fused) | GFLOPs @640 (Ultralytics summary) | GFLOPs @416 (measured) | `best.pt` size |
|---|---|---|---|---|---|
| YOLOv9t | 544 / 198 | 2,005,603 / 2,074,803 | 7.8 (fused 8.1) | 3.54 | 4,815,433 B (4.82 MB) |
| YOLOv12n | 272 / 160 | 2,568,243 / 2,571,627 | 7.5 (fused 7.4) | 2.94 | 5,520,538 B (5.52 MB) |

The fused YOLOv9t count is higher than the unfused count because Ultralytics re-parameterises its RepConv/ELAN blocks during fusion. The figures are reported exactly as Ultralytics printed them.

---

## Part 4 — Training data: the 594-image half subset

### 4.1 Dataset audit (original splits, read-only)

| Split | Images | Labels | Orphan labels | Label format problems | Image size |
|---|---|---|---|---|---|
| train | 1,191 | 1,191 | 0 | 0 | 416 × 416 (all) |
| valid | 116 | 116 | 0 | 0 | 416 × 416 (all) |
| test | 55 | 55 | 0 | 0 | 416 × 416 (all) |

All label lines have 5 values, class 0 and normalised coordinates in [0, 1]. The training split has 10,002 boxes, which agrees with the correction in Astley's audit (Part 4.4).

### 4.2 Finding: Roboflow augmentation groups

- `README.roboflow.txt` states that each training source image was exported as **3 augmented versions** (brightness ±26%, 6% salt-and-pepper noise).
- Grouping training files by the stem before `.rf.` gives **375 groups**: 353 triplets and 22 groups of 6. Pixel comparison shows the 6-groups are two different scenes that share a name.
- Sampling 595 individual files at random keeps about 87% of the unique scenes, so it would not really halve the information in the training data.
- **The subset here therefore samples whole groups.**

### 4.3 Finding: training scene names that also appear in valid/test

- **10 training groups** share a source name with a validation image (8 groups) or a test image (2 groups):
  - `CLBJ_035_2019`
  - `DEJU_010_2019`, `DEJU_011_2019`
  - `DELA_010_2018`, `DELA_011_2019`, `DELA_012_2019`
  - `DSNY_002_2019`, `DSNY_005_2018`, `DSNY_006_2018`, `DSNY_006_2019`
- The mean absolute pixel difference to the closest training version is 9.7–46.1 grey levels, against 49.0 for two unrelated images. Some of these are therefore probably the same scene with augmentation applied.
- They were **excluded from the subset** as a precaution. The original splits were left unchanged.
- YOLOv5n and YOLOv7-tiny may have been trained with these images, because their subset sampled individual files.

### 4.4 Selection procedure (`comparison_models/common/make_subset.py`)

1. Group the training files by source stem and drop the 10 overlapping groups (365 groups, 1,161 files eligible).
2. Stratify by NEON site code (15 sites). In each site, sort the groups, shuffle them with `random.Random(0)`, and take groups until the site's image count is closest to half of its original count.
3. Whole groups undershoot each site's target, so a seeded top-up pass adds leftover groups until the total is closest to 0.5 × 1,191 = 595.5.
4. Write `subset_manifest.txt` (594 repo-relative paths), `subset_stats.json` and `subset_stats.md`.
5. Running the script twice gives an identical manifest (md5 `c076770420d14f85effe991b738aeb30`). Assertions confirm that no validation or test file or group is selected, and that every selected image has a label.

**Difference from Astley's 595-image subset.** The two subsets are different sets of images. A cross-check could not be run because Astley's subset file is not on this machine.

| | YOLOv5n / YOLOv7-tiny subset | YOLOv9t / YOLOv12n subset |
|---|---|---|
| Size | 595 | 594 |
| Seed | 42 | 0 |
| Unit sampled | individual files (`rng.sample`) | Roboflow augmentation groups, stratified by site |
| Held-out-overlap groups | not excluded | excluded (10 groups) |
| Boxes | 4,939 | 5,190 |
| Background images | 19 | 15 |
| Mean trees per image | 8.30 | 8.74 |

### 4.5 Subset statistics (from `subset_stats.md`)

| Statistic | Full train | 594 subset |
|---|---|---|
| Images / source groups | 1,191 / 375 | 594 / 188 |
| Tree instances | 10,002 | 5,190 |
| Images without trees | 39 | 15 |
| Instances per image: mean / median / p90 / max | 8.40 / 7 / 16 / 31 | 8.74 / 8 / 16 / 31 |
| Median box width × height (px) | 55 × 53 | 54 × 53 |
| Box area p10 / median / p90 (px²) | 552 / 3,048 / 9,433 | 572 / 2,912 / 9,375 |
| Share of boxes small (< 32²) / medium / large (≥ 96²) | 23.6% / 65.8% / 10.6% | 24.7% / 64.9% / 10.4% |

Every site keeps about half its images (for example BART 174 → 90, DEJU 201 → 99, BLAN 15 → 6).

### 4.6 How the data is loaded without modifying the original dataset

- Training reads `comparison_models/data/tree_top_view_half/` (git-ignored). This contains **per-file symlinks** to the original images and labels: 594 train, 116 validation and 55 test.
- Its `data.yaml` points `val` and `test` at the original, unchanged splits.
- Ultralytics writes its `labels.cache` inside this view. This avoids a problem Astley's audit records (Part 14.2, discrepancy 4), where the subset cache **overwrote `datasets/Tree-Top-View/train/labels.cache`**.

---

## Part 5 — Training configuration (from `runs/*/args.yaml`)

| Setting | YOLOv9t | YOLOv12n |
|---|---|---|
| Data | 594-image subset | 594-image subset |
| imgsz / batch | 416 / 8 | 416 / 8 |
| epochs / patience | 30 / 5, overridden by `time: 0.5` | 30 / 5, overridden by `time: 0.5` |
| Epochs actually run | **17** (plan re-estimated 30 → 35 → 24 → 21 → 20 → 19 → 18 → 17) | **17** (plan re-estimated 30 → 18 → 17) |
| seed / deterministic | 0 / true | 0 / true |
| optimizer | auto → **AdamW lr 0.002, momentum 0.9** (log confirmed) | auto → **AdamW lr 0.002, momentum 0.9** (log confirmed) |
| lrf / weight_decay / warmup_epochs | 0.01 / 0.0005 / 3.0 | 0.01 / 0.0005 / 3.0 |
| Loss gains box / cls / dfl | 7.5 / 0.5 / 1.5 | 7.5 / 0.5 / 1.5 |
| Augmentation | hsv 0.015/0.7/0.4, translate 0.1, scale 0.5, fliplr 0.5, mosaic 1.0, mixup 0, erasing 0.4 | same |
| close_mosaic | 10 → **mosaic off from epoch 9** (after 8 of 17) | 10 → **mosaic off from epoch 8** (after 7 of 17) |
| workers / device | 2 / CPU, 8 threads | 2 / CPU, 8 threads |
| Concurrency | run at the same time as YOLOv12n | run at the same time as YOLOv9t |

**Effect of the time budget.**
- Ultralytics' `time` argument re-estimates the total number of epochs after every epoch, and rescales the learning-rate schedule so that it decays fully by the end of the budget.
- Both runs therefore finished with a fully decayed learning rate (`lr/pg0` = 0.0001 at epoch 17), rather than being cut off mid-schedule.
- Because the plan shrank to 17–18 epochs, `close_mosaic=10` turned mosaic off much earlier than it would in a 30-epoch run.

---

## Part 6 — Epoch-by-epoch training progression (`results.csv`)

### 6.1 YOLOv9t

| Ep | Time (s) | Train box | Train cls | Train dfl | Val P | Val R | Val mAP50 | Val mAP50-95 | Val box | Val cls | Val dfl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 51.5 | 2.2934 | 2.6890 | 1.9555 | 0.2851 | 0.3112 | 0.2068 | 0.0766 | 2.3033 | 2.6743 | 2.3677 |
| 2 | 151.0 | 1.9497 | 2.0376 | 1.6833 | 0.2813 | 0.3455 | 0.2428 | 0.0869 | 2.0543 | 3.0320 | 2.0990 |
| 3 | 262.2 | 1.8633 | 1.9074 | 1.6284 | 0.3071 | 0.4469 | 0.2939 | 0.1105 | 1.9865 | 2.5620 | 1.8284 |
| 4 | 376.6 | 1.8783 | 1.8467 | 1.5944 | 0.3869 | 0.4174 | 0.3576 | 0.1591 | 1.8138 | 2.1127 | 1.6882 |
| 5 | 482.6 | 1.7999 | 1.7408 | 1.5610 | 0.4482 | 0.4506 | 0.4354 | 0.2112 | 1.7433 | 1.9471 | 1.6556 |
| 6 | 592.2 | 1.7765 | 1.7537 | 1.5405 | 0.3849 | 0.4228 | 0.3633 | 0.1658 | 1.8214 | 1.9079 | 1.6405 |
| 7 | 698.5 | 1.7414 | 1.7095 | 1.5005 | 0.4415 | 0.4554 | 0.4355 | 0.1985 | 1.7352 | 1.7455 | 1.5931 |
| 8 | 812.5 | 1.7418 | 1.6776 | 1.5138 | 0.3636 | 0.3831 | 0.3279 | 0.1494 | 1.8199 | 2.0331 | 1.6591 |
| 9 | 930.5 | 1.7119 | 1.7349 | 1.5143 | 0.4347 | 0.4635 | 0.4091 | 0.1858 | 1.8181 | 1.9304 | 1.6680 |
| 10 | 1,047.3 | 1.6527 | 1.6171 | 1.4895 | 0.4397 | 0.5107 | 0.4672 | 0.2316 | 1.7061 | 1.7329 | 1.5721 |
| 11 | 1,161.8 | 1.6422 | 1.5656 | 1.4742 | 0.4689 | 0.5107 | 0.4966 | 0.2426 | 1.6792 | 1.6032 | 1.5918 |
| 12 | 1,275.1 | 1.6149 | 1.5182 | 1.4740 | 0.4170 | 0.4850 | 0.4573 | 0.2192 | 1.7253 | 1.6601 | 1.5836 |
| 13 | 1,388.3 | 1.5951 | 1.4892 | 1.4642 | 0.4196 | 0.4894 | 0.4383 | 0.2123 | 1.6667 | 1.6124 | 1.5675 |
| 14 | 1,494.8 | 1.5613 | 1.4808 | 1.4243 | 0.4173 | **0.5429** | 0.4763 | 0.2404 | 1.6604 | 1.6032 | 1.5641 |
| 15 | 1,605.5 | 1.5356 | 1.4359 | 1.4193 | 0.4745 | 0.4989 | 0.5055 | 0.2596 | 1.6610 | 1.5005 | 1.5640 |
| 16 | 1,713.9 | 1.5171 | 1.3914 | 1.4231 | 0.5121 | 0.5107 | **0.5259** | 0.2715 | **1.6394** | **1.4502** | **1.5585** |
| **17 (best + final)** | 1,815.3 | **1.4820** | **1.3461** | **1.3959** | **0.5268** | 0.4818 | 0.5188 | **0.2720** | 1.6559 | 1.4813 | 1.5764 |

Epoch 1 ran alone (51.5 s). From epoch 2 onward YOLOv12n trained at the same time, and epochs took about 105–118 s.

**`best.pt` is epoch 17.** In ultralytics 8.4.177, checkpoint fitness is mAP50-95 alone (`DetMetrics.fitness` weights `[0, 0, 0, 1]`). The stored fitness in `best.pt` is 0.27203, which is epoch 17's mAP50-95 and beats epoch 16 (0.27146), even though epoch 16 has the higher mAP50 (0.5259 vs 0.5188). The first version of `train_summary.json` used the older 0.1·mAP50 + 0.9·mAP50-95 formula and reported epoch 16. This was corrected (Part 12, item 9).

### 6.2 YOLOv12n

| Ep | Time (s) | Train box | Train cls | Train dfl | Val P | Val R | Val mAP50 | Val mAP50-95 | Val box | Val cls | Val dfl |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100.3 | 2.2431 | 2.6672 | 1.8011 | 0.3240 | 0.3646 | 0.2691 | 0.1069 | 2.0452 | 4.4142 | 1.8794 |
| 2 | 207.5 | 1.9574 | 1.9953 | 1.5965 | 0.3443 | 0.3777 | 0.2708 | 0.1008 | 2.0599 | 2.5153 | 1.8405 |
| 3 | 319.1 | 1.9035 | 1.9143 | 1.5643 | 0.3223 | 0.3101 | 0.2456 | 0.0888 | 2.1507 | 2.2658 | 1.8900 |
| 4 | 428.6 | 1.8792 | 1.8584 | 1.5370 | 0.3057 | 0.4356 | 0.2746 | 0.1114 | 1.9883 | 3.4666 | 1.7428 |
| 5 | 537.6 | 1.8031 | 1.7369 | 1.4886 | 0.4172 | 0.4678 | 0.4140 | 0.2071 | 1.7408 | 1.8823 | 1.5553 |
| 6 | 644.8 | 1.8031 | 1.7449 | 1.4875 | 0.4602 | 0.4367 | 0.4325 | 0.1881 | 1.8802 | 1.7769 | 1.6229 |
| 7 | 758.9 | 1.7683 | 1.7159 | 1.4605 | 0.3779 | 0.4989 | 0.3833 | 0.1739 | 1.7391 | 2.0417 | 1.4920 |
| 8 | 871.1 | 1.7638 | 1.7454 | 1.4759 | 0.4018 | 0.5161 | 0.4336 | 0.1945 | 1.7180 | 1.7184 | 1.4959 |
| 9 | 979.0 | 1.7333 | 1.6769 | 1.4785 | 0.4921 | 0.4959 | 0.4741 | 0.2231 | 1.7580 | 1.5923 | 1.5254 |
| 10 | 1,088.1 | 1.6798 | 1.5965 | 1.4333 | 0.4158 | 0.4571 | 0.3991 | 0.1874 | 1.7217 | 1.8421 | 1.5314 |
| 11 | 1,190.7 | 1.6557 | 1.5656 | 1.4307 | 0.4754 | 0.4890 | 0.4960 | 0.2425 | 1.7087 | 1.5781 | 1.5289 |
| 12 | 1,307.5 | 1.6383 | 1.5321 | 1.4166 | 0.4484 | 0.5215 | 0.4972 | 0.2289 | 1.7166 | 1.5335 | 1.4950 |
| 13 | 1,413.0 | 1.6146 | 1.5021 | 1.4151 | 0.4769 | 0.5032 | 0.4945 | 0.2369 | 1.7103 | 1.5703 | 1.4954 |
| 14 | 1,513.6 | 1.5863 | 1.4713 | 1.3856 | 0.4404 | 0.4880 | 0.4378 | 0.2036 | 1.6960 | 1.6217 | 1.4934 |
| 15 | 1,611.2 | 1.5860 | 1.4453 | 1.3821 | 0.5132 | 0.5236 | 0.5317 | 0.2607 | 1.6561 | 1.4618 | 1.4680 |
| **16 (best)** | 1,717.0 | **1.5123** | 1.3662 | 1.3512 | **0.5311** | **0.5440** | **0.5527** | **0.2759** | **1.6214** | **1.4007** | **1.4609** |
| 17 (final) | 1,789.5 | 1.5221 | **1.3324** | **1.3343** | 0.5234 | 0.5375 | 0.5512 | 0.2738 | 1.6426 | 1.4082 | 1.4629 |

The very high validation classification loss at epochs 1 and 4 (4.41 and 3.47) comes from the freshly initialised single-class head during warm-up. It settles by epoch 5.

---

## Part 7 — Validation and held-out test results

All evaluation used the **best checkpoint** (`runs/<run>/weights/best.pt`: epoch 17 for YOLOv9t, epoch 16 for YOLOv12n) and the Ultralytics validator (conf 0.001, NMS IoU 0.7, batch 8, 416 px, CPU). The test split was evaluated **exactly once per model**, after training was final. Checkpoint selection used validation only.

| Split | Model | Precision | Recall | F1 | mAP@0.50 | mAP@0.50:0.95 |
|---|---|---|---|---|---|---|
| Validation (116 imgs, 932 GT) | YOLOv9t | 0.5294 | 0.4839 | 0.5056 | 0.5188 | 0.2719 |
| | YOLOv12n | 0.5309 | 0.5440 | 0.5374 | 0.5525 | 0.2761 |
| Test (55 imgs, 443 GT) | YOLOv9t | 0.5052 | 0.5532 | 0.5281 | 0.5256 | 0.2640 |
| | YOLOv12n | 0.5219 | 0.5421 | 0.5318 | 0.5383 | 0.2596 |

---

## Part 8 — Tree-counting evaluation (count = boxes at conf ≥ 0.25, same rule as `src/evaluate_counting.py`)

| Split | Model | GT | Predicted | Net | MAE | RMSE | Bias | Exact | Within ±1 | Within ±2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Validation | YOLOv9t | 932 | n/a | n/a | 3.3966 | n/a | n/a | n/a | n/a | n/a |
| | YOLOv12n | 932 | 855 | −77 | 2.9914 | n/a | n/a | n/a | n/a | n/a |
| Test | YOLOv9t | 443 | 428 | −15 (−3.39%) | **2.8545** | 4.0430 | −0.2727 | 16.4% | 40.0% | 58.2% |
| | YOLOv12n | 443 | 394 | −49 (−11.06%) | 3.0364 | **4.0204** | −0.8909 | 16.4% | 38.2% | 52.7% |

Per-image counts are in `yolov9/metrics/test_counting.csv` and `yolov12/metrics/test_counting.csv`. MAE broken down by density and by site is in `REPORT.md` §3.

Breaking test MAE down by tree density:

| Model | 0–5 trees (20 imgs) | 6–10 (19) | 11+ (16) |
|---|---|---|---|
| YOLOv9t | 1.50 | 2.63 | 4.81 |
| YOLOv12n | 1.90 | 3.00 | 4.50 |

As Astley's audit explains (Part 5.6), the per-image counting error equals |FPᵢ − FNᵢ|, so false positives and false negatives in the same image cancel. This is how YOLOv9t can have the lower MAE while having the lower mAP@0.50 of the two models.

---

## Part 9 — Error analysis: TP / FP / FN

The algorithm is identical to the project's existing `src/analyze_predictions*.py` scripts: integer pixel boxes, predictions sorted by confidence, greedy one-to-one matching at IoU ≥ 0.50, conf ≥ 0.25. It was implemented in `common/report_details.py`.

| Subset | Model | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|---|
| First 30 sorted validation images (270 GT), the project's standard error-analysis subset | YOLOv9t | 152 | 117 | 118 | 0.5651 | 0.5630 | 0.5640 |
| | YOLOv12n | **159** | **109** | **111** | **0.5933** | **0.5889** | **0.5911** |
| Full test split (55 images, 443 GT) | YOLOv9t | 226 | 202 | 217 | 0.5280 | 0.5102 | 0.5189 |
| | YOLOv12n | 224 | **170** | 219 | **0.5685** | 0.5056 | **0.5352** |

---

## Part 10 — Inference timing

Two protocols were run, matching the two protocols in Astley's audit (Part 12.3). Both ran on the i5-12500H with no other job running.

| Model | Protocol A: `batch=1`, 90 passes (30 test images × 3 after 5 warm-ups), pure inference | Protocol A total | Protocol B: Ultralytics test validator `batch=8`, preprocess / inference / postprocess | Protocol B total |
|---|---|---|---|---|
| YOLOv9t | **21.50 ± 0.45 ms** | 22.20 ± 0.47 ms (about 45 FPS) | 0.50 / 19.67 / 1.06 ms | 21.23 ms |
| YOLOv12n | **21.19 ± 0.58 ms** | 21.88 ± 0.58 ms (about 46 FPS) | 0.49 / 19.06 / 1.25 ms | 20.80 ms |

**These timings cannot be compared with YOLOv5n and YOLOv7-tiny.** Those models were timed on an i7-1255U running Windows with a different torch and ultralytics build. Only YOLOv9t against YOLOv12n is a like-for-like comparison here, and the two are effectively tied. A fair latency table needs all checkpoints timed on one machine.

---

## Part 11 — Consolidated comparison of the half-subset models (test split, 55 images, 443 trees)

| Model | Train data | Epochs | Train time | P | R | mAP@0.50 | mAP@0.50:0.95 | Count MAE | Predicted trees | Params | `best.pt` | Source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| YOLOv5n (`yolov5nu`) | 595, seed 42 | 30 | 1.94 h | **0.5994** | 0.5485 | **0.5778** | **0.2807** | 3.2000 | 449 (**+6**) | 2.50 M | 5.00 MB | Astley audit |
| YOLOv7-tiny | 595, seed 42 | 30 | 2.24 h | 0.5635 | 0.5440 | 0.5460 | 0.2519 | 3.7636 | 564 (+121) | 6.01 M | 11.68 MB | Astley audit |
| **YOLOv9t** | 594, seed 0 (groups) | **17 (0.5 h budget)** | **0.51 h** | 0.5052 | **0.5532** | 0.5256 | 0.2640 | **2.8545** | 428 (−15) | **2.01 M** | **4.82 MB** | this audit |
| **YOLOv12n** | 594, seed 0 (groups) | **17 (0.5 h budget)** | **0.50 h** | 0.5219 | 0.5421 | 0.5383 | 0.2596 | 3.0364 | 394 (−49) | 2.57 M | 5.52 MB | this audit |

**What can and cannot be concluded**
- **Within the 594-subset pair (YOLOv9t vs YOLOv12n):** controlled. Same data, settings, time budget, software and machine.
  - YOLOv12n is better on validation detection (mAP50 0.553 vs 0.519), test precision, test F1 and false positives.
  - YOLOv9t is better on counting MAE (2.85 vs 3.04), is smaller, and has less count bias (−15 vs −49 trees).
- **Against YOLOv5n and YOLOv7-tiny:** not controlled. Different subset (Part 4.4), 17 vs 30 epochs, a different machine, and ultralytics 8.4.177 vs 8.4.149 (YOLOv7-tiny also used a different framework).
- **Across the four half-subset models:** YOLOv5n has the best test detection (P, mAP@0.50, mAP@0.50:0.95). YOLOv9t has the lowest counting MAE. YOLOv9t and YOLOv12n reached their scores with 17 epochs against 30.

---

## Part 12 — Errors, warnings and troubleshooting

| # | Component | Problem observed | Root cause | Resolution | Effect on results |
|---|---|---|---|---|---|
| 1 | Environment | Project `.venv` existed but was empty | An earlier install command was interrupted by the user after `uv venv` had already run | Kept and reused; packages installed later | None |
| 2 | Environment | CUDA PyTorch download stalled (about 1 MB/s falling to 0.05 MB/s; 3.5 GB of about 4.5 GB done after 2 h) | Network throughput | Cancelled and switched to the CPU wheel (`--index-url …/whl/cpu`, torch 2.14.1+cpu), which took about 70 min | Training on CPU instead of GPU, which led to the time budget |
| 3 | Auto-launcher | The background launcher never started training (about 35 min lost) | It waited on `pgrep -f "uv pip install"`, which matched its own command line, so it waited forever | Killed and replaced with a direct launch | Delay only |
| 4 | Training launch | YOLOv12n crashed at start with `FileExistsError: …/tree_top_view_half/train/images` | Both processes built the shared dataset view at the same moment (a race condition) | The view was verified complete (594/116/55) and YOLOv12n relaunched about 1 min later. `build_data_view()` was then changed to build in a per-process temp directory with an atomic `os.replace` | YOLOv12n started 1 min after YOLOv9t; data identical |
| 5 | `train_summary.json` | Recorded `early_stopped: true` | `epochs_done < epochs` was treated as early stopping, but the stop came from the time budget | Replaced with `stop_reason: "time budget (0.5 h)"` in code and both JSON files | Labelling only |
| 6 | Latency JSON | Hardware recorded as `x86_64` | `platform.processor()` is empty on Linux | Reads `/proc/cpuinfo` instead; both JSON files corrected to "12th Gen Intel(R) Core(TM) i5-12500H" | Labelling only |
| 7 | Mosaic schedule | Mosaic closed after epoch 8 (YOLOv9t) and after epoch 7 (YOLOv12n) | `close_mosaic=10` applies to the re-planned 17–18-epoch run | Documented (Part 5); not changed after the fact | Training dynamics differ from the 30-epoch runs |
| 8 | Ultralytics | `optimizer=auto found, ignoring lr0=0.01` | Expected Ultralytics behaviour with `optimizer=auto` | n/a | AdamW lr 0.002 for both models |
| 9 | `train_summary.json`, `REPORT.md` | YOLOv9t best epoch reported as 16 | The summary code used the 0.1·mAP50 + 0.9·mAP50-95 fitness from older Ultralytics; 8.4.x uses mAP50-95 only. Confirmed from the fitness stored in `best.pt` (0.27203 = epoch 17) | `_best_epoch_from_results()` now uses mAP50-95; JSON and reports corrected to epoch 17 | Labelling only; all evaluations used the real `best.pt` |

---

## Part 13 — Documentation reconciliation

1. **Best-epoch numbers.**
   - In-training validation at the best epoch (17) for YOLOv9t was P 0.5268, R 0.4818, mAP50 0.5188, mAP50-95 0.2720.
   - Standalone re-validation of `best.pt` (`val_metrics.json`) gave P 0.5294, R 0.4839, mAP50 0.5188, mAP50-95 0.2719.
   - For YOLOv12n the two sets are 0.5311 / 0.5440 / 0.5527 / 0.2759 (in training) against 0.5309 / 0.5440 / 0.5525 / 0.2761 (standalone).
   - The small shift comes from FP16 checkpoint storage and layer fusion at reload (also described in Astley's audit, Part 14.2, item 3). The reports use the standalone numbers.
2. **Parameter counts.**
   - `report_details.json` counts every tensor in the loaded model: YOLOv9t 2,109,523 and YOLOv12n 2,583,091.
   - Ultralytics' printed summary gives 2,005,603 and 2,568,243 unfused. This audit cites the Ultralytics summary, and `REPORT.md` §5 cites the `report_details.json` figures (2.11 M and 2.58 M).
   - Both are correct; they count slightly different things (DFL/anchor buffers, fused BN).
3. **YOLOv5n and YOLOv7-tiny numbers** in Part 11 are quoted from Astley's audit (Parts 9.4, 10.6 and 12.2).
4. **Training durations.**
   - YOLOv9t: 1,833.9 s (script wall-clock) against 1,815.3 s (internal epoch timer, 0.504 h in log).
   - YOLOv12n: 1,798.8 s (wall-clock) against 1,789.5 s (internal, 0.497 h).

---

## Part 14 — Git status and repository hygiene

- **Branch:** `yolov12-yolov9-comparison`.
  - Commit `62b4dce` ("Add YOLOv9t and YOLOv12n comparison models") is pushed to `origin`.
  - It contains 32 files: `.gitignore` plus `comparison_models/` code, docs, manifest and metrics.
- **Follow-up commit** (this audit, `REPORT.md`, `report_details.py`, `report_details.json`): staged explicitly by path, never with `git add .`.
- **Never committed:**
  - `*.pt`
  - `comparison_models/*/runs/`, `*/weights/`, `*/predictions/`, `comparison_models/data/`
  - `.venv/`, `*.cache`, `context.md`
- **`.gitignore` additions:** `comparison_models/data/`, `comparison_models/*/weights/`, `comparison_models/*/runs/`, `comparison_models/*/predictions/`, and `comparison_models/external_dataset/*` (with `!comparison_models/external_dataset/README.md`).
- **Merging with Astley's working tree.** Astley's `comparison_models/` (YOLOv5n, YOLOv7-tiny, `testdataset100`) is untracked on Astley's machine and has a different layout (`prepare_subset.py` at the top level, `yolov5/`, `yolov7_tiny/`). The two layouts do not collide on file names except `.gitignore`, which needs a manual merge. Astley's `.gitignore` adds `runs/`, `outputs/`, `*.npy` and the `testdataset100/` rules.

---

## Part 15 — Reproducibility

```bash
# environment (Linux)
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision
uv pip install --python .venv/bin/python ultralytics==8.4.177

# subset (deterministic; md5 c076770420d14f85effe991b738aeb30)
python3 comparison_models/common/make_subset.py

# training (as run: both at once, 8 threads each, 0.5 h budget)
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python comparison_models/yolov9/train.py  --device cpu --workers 2 --time 0.5
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python comparison_models/yolov12/train.py --device cpu --workers 2 --time 0.5
# full protocol instead: drop --time (30 epochs, patience 5)

# evaluation (validation first; test once)
.venv/bin/python comparison_models/yolov9/evaluate.py --split val --skip-latency
.venv/bin/python comparison_models/yolov9/evaluate.py --split test
.venv/bin/python comparison_models/yolov12/evaluate.py --split val --skip-latency
.venv/bin/python comparison_models/yolov12/evaluate.py --split test

# extra statistics and summary tables
.venv/bin/python comparison_models/common/report_details.py
.venv/bin/python comparison_models/common/summarize.py

# single image / folder counting
.venv/bin/python comparison_models/yolov9/predict.py datasets/Tree-Top-View/test/images/<image>.jpg
```

---

## Part 16 — External evaluation: SelvaBox `testdataset100` (zero-shot)

- **Dataset.** Astley's `testdataset100.zip` (424.6 MB, 208 files) was checked before extraction: no absolute or `..` paths. It was extracted to `comparison_models/external_dataset/testdataset100/` (git-ignored).
  - All 12 of Astley's validation checks are recorded as PASS.
  - It contains 100 PNG tiles of 1777 × 1777 px (4.5 cm GSD, 80 m × 80 m) and 8,111 crowns: Brazil ZF2 13 tiles / 1,641 crowns, Ecuador Tiputini 60 / 1,095, Panama Agua Salud 27 / 5,375.
- **Loader.** The dataset stores labels in `annotations/`, so `comparison_models/common/evaluate_external.py` builds a symlink view with an `images/` + `labels/` layout in `comparison_models/data/testdataset100_view/`. The extracted files are only read.
- **Pre-specified settings** (fixed before any result was seen; nothing was tuned on this set):

  | Setting | Value | Reason |
  |---|---|---|
  | imgsz (primary) | 832 | Training ground resolution ≈ 9.6 cm/px (NEON 10 cm tiles, 400 → 416 px); 80 m ÷ 0.096 ≈ 833 → 832 (stride 32) |
  | imgsz (secondary) | 416 | Plain training input size |
  | conf for counting and TP/FP/FN | 0.25 | Same as the internal test split |
  | max_det | 1,000 | Up to 361 crowns per tile; the Ultralytics default (300) would cap counts |
  | Ultralytics validator | conf 0.001, NMS IoU 0.7, batch 4, CPU | Standard |

- **Results** (`yolov9/metrics/external_testdataset100_imgsz{832,416}.json/.csv`, `yolov12/metrics/…`):

  | Model | imgsz | P | R | mAP@0.50 | mAP@0.50:0.95 | Predicted / 8,111 | MAE | RMSE | Bias | TP / FP / FN @0.25 | Inference ms/tile |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | YOLOv9t | 832 | 0.4556 | 0.2743 | 0.2593 | 0.1066 | 1,450 (−82.1%) | 66.77 | 105.30 | −66.61 | 968 / 482 / 7,143 | 70.6 |
  | YOLOv12n | 832 | 0.4125 | 0.2540 | 0.2399 | 0.0990 | 1,748 (−78.4%) | 63.91 | 104.56 | −63.63 | 1,066 / 682 / 7,045 | 80.4 |
  | YOLOv9t | 416 | 0.4897 | 0.1537 | 0.1524 | 0.0579 | 670 (−91.7%) | 74.47 | 114.26 | −74.41 | 435 / 235 / 7,676 | 23.9 |
  | YOLOv12n | 416 | 0.5145 | 0.1634 | 0.1823 | 0.0722 | 727 (−91.0%) | 74.02 | 114.16 | −73.84 | 464 / 263 / 7,647 | 23.9 |

- **By site at 832 px:**

  | Site | Images | GT | YOLOv9t pred / MAE / recall@0.25 | YOLOv12n pred / MAE / recall@0.25 |
  |---|---|---|---|---|
  | Ecuador Tiputini | 60 | 1,095 | 490 / 10.35 / 0.232 | 709 / 6.90 / 0.304 |
  | Brazil ZF2 | 13 | 1,641 | 378 / 97.15 / 0.171 | 444 / 92.08 / 0.189 |
  | Panama Agua Salud | 27 | 5,375 | 582 / 177.52 / 0.081 | 595 / 177.04 / 0.079 |

- **Crown-size check** (from the SelvaBox labels, √(w·h) in metres): p10 / median / p90 = 2.2 / 5.0 / 10.2 m. Tree-Top-View training crowns are about 2.3 / 5.1 / 9.3 m. At 832 px the crowns are therefore at the scale the models were trained on.
  - The failure comes from **canopy type and density**: closed tropical canopy with up to 361 touching crowns per tile, against about 8.4 trees per tile in open NEON woodland.
  - Recall collapses (Panama about 8%) while precision at conf 0.25 stays at 0.61–0.67.
- **Conclusion.** Neither model transfers to dense tropical forest without fine-tuning. Between them, YOLOv9t has slightly better detection metrics and fewer false positives, and YOLOv12n finds more trees and counts slightly better. The difference is small compared with the domain gap.

---

## Part 17 — Remaining work

1. **Same-settings external results for YOLOv5n and YOLOv7-tiny.** For a like-for-like external comparison, Astley should evaluate those models on `testdataset100` with the same settings: imgsz 832 (and 416), conf 0.25, max_det 1000, the greedy IoU 0.5 matching.
2. **Optional fair re-run.** If time allows, retrain YOLOv9t and YOLOv12n for the full 30 epochs, one at a time, and ideally on Astley's seed-42 subset. That would make the comparison with YOLOv5n and YOLOv7-tiny controlled.
3. **Optional single-machine latency.** Time every `best.pt` file on one machine with Protocol A.
