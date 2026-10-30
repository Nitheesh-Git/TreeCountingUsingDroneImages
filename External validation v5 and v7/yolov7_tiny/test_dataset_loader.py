import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR / "yolov7_repo"
SHIMS_DIR = SCRIPT_DIR / "shims"

sys.path.insert(0, str(SHIMS_DIR))
sys.path.insert(0, str(REPO_DIR))

import yaml
from utils.datasets import create_dataloader
from utils.general import check_dataset

data_cfg = SCRIPT_DIR / "data_subset_595_yolov7.yaml"
with open(data_cfg, "r") as f:
    data_dict = yaml.safe_load(f)

hyp_path = REPO_DIR / "data" / "hyp.scratch.tiny.yaml"
with open(hyp_path, "r") as f:
    hyp = yaml.safe_load(f)

class Opt:
    single_cls = False
    rect = False
    cache_images = False
    image_weights = False
    quad = False
    world_size = 1
    workers = 0

opt = Opt()
train_loader, train_dataset = create_dataloader(
    data_dict["train"], 416, 8, 32, opt, hyp=hyp, augment=True, workers=0
)
val_loader, val_dataset = create_dataloader(
    data_dict["val"], 416, 16, 32, opt, hyp=hyp, augment=False, rect=True, pad=0.5, workers=0
)

print(f"[SUCCESS] Train images loaded: {len(train_dataset)} (batches: {len(train_loader)})")
print(f"[SUCCESS] Val images loaded  : {len(val_dataset)} (batches: {len(val_loader)})")
