import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR / "yolov7_repo"
SHIMS_DIR = SCRIPT_DIR / "shims"

def create_shims():
    SHIMS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. tqdm shim with clean progress output
    tqdm_code = '''import sys
import time

class tqdm:
    def __init__(self, iterable=None, desc='', total=None, initial=0, bar_format=None, **kwargs):
        self.iterable = iterable
        self.desc = desc or ''
        self.total = total if total is not None else (len(iterable) if hasattr(iterable, '__len__') else None)
        self.n = initial
        self.start_time = time.time()
        self.last_print_time = 0.0
        if iterable is None and self.desc:
            print(self.desc, flush=True)

    def __iter__(self):
        if self.iterable is None:
            return
        for idx, item in enumerate(self.iterable):
            self.n = idx + 1
            now = time.time()
            if idx == 0 or self.n == self.total or (now - self.last_print_time) >= 4.0:
                self.last_print_time = now
                elapsed = max(now - self.start_time, 1e-6)
                rate = self.n / elapsed
                if self.total:
                    pct = 100.0 * self.n / self.total
                    rem = (self.total - self.n) / max(rate, 1e-6)
                    print(f"{self.desc} [{self.n}/{self.total} ({pct:.0f}%) | {1.0/max(rate,1e-6):.2f}s/it | {elapsed:.1f}s<{rem:.1f}s]", flush=True)
                else:
                    print(f"{self.desc} [{self.n} it | {elapsed:.1f}s]", flush=True)
            yield item

    def set_description(self, desc=None, refresh=True):
        self.desc = desc or ''

    def update(self, n=1):
        self.n += n

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
'''
    (SHIMS_DIR / "tqdm.py").write_text(tqdm_code, encoding="utf-8")

    # 2. tensorboard shim
    tb_dir = SHIMS_DIR / "tensorboard"
    tb_dir.mkdir(exist_ok=True)
    (tb_dir / "__init__.py").write_text("# dummy tensorboard\n", encoding="utf-8")

    # 3. scipy shim
    scipy_dir = SHIMS_DIR / "scipy"
    scipy_dir.mkdir(exist_ok=True)
    (scipy_dir / "__init__.py").write_text("", encoding="utf-8")
    (scipy_dir / "signal.py").write_text(
        "def butter(*a, **k): return [1.0], [1.0]\n"
        "def filtfilt(b, a, data, *args, **kwargs): return data\n",
        encoding="utf-8"
    )
    cluster_dir = scipy_dir / "cluster"
    cluster_dir.mkdir(exist_ok=True)
    (cluster_dir / "__init__.py").write_text("", encoding="utf-8")
    (cluster_dir / "vq.py").write_text(
        "import numpy as np\n"
        "def kmeans(obs, k_or_guess, iter=20, thresh=1e-5):\n"
        "    idx = np.linspace(0, len(obs)-1, k_or_guess, dtype=int)\n"
        "    return obs[idx], 0.0\n",
        encoding="utf-8"
    )
    ndimage_dir = scipy_dir / "ndimage"
    ndimage_dir.mkdir(exist_ok=True)
    (ndimage_dir / "__init__.py").write_text("def gaussian_filter1d(x, *a, **k): return x\n", encoding="utf-8")
    (ndimage_dir / "filters.py").write_text("def gaussian_filter1d(x, *a, **k): return x\n", encoding="utf-8")

    # 4. pandas & seaborn shims
    (SHIMS_DIR / "pandas.py").write_text(
        "class Display:\n"
        "    max_columns = 10\n"
        "    width = 100\n"
        "class Options:\n"
        "    display = Display()\n"
        "options = Options()\n"
        "class DataFrame:\n"
        "    def __init__(self, *a, **k): pass\n"
        "    def to_csv(self, *a, **k): pass\n"
        "def read_csv(*a, **k): return DataFrame()\n",
        encoding="utf-8"
    )
    (SHIMS_DIR / "seaborn.py").write_text(
        "def PairGrid(*a, **k): return None\n"
        "def heatmap(*a, **k): return None\n",
        encoding="utf-8"
    )
    print(f"[OK] Created pure-Python shims in {SHIMS_DIR}")

def patch_repo():
    compat_header = (
        "# --- PY313_PT2_COMPAT_PATCH ---\n"
        "import numpy as _np\n"
        "if not hasattr(_np, 'int'): _np.int = int\n"
        "if not hasattr(_np, 'float'): _np.float = float\n"
        "if not hasattr(_np, 'bool'): _np.bool = bool\n"
        "if not hasattr(_np, 'object'): _np.object = object\n"
        "if not hasattr(_np, 'str'): _np.str = str\n"
        "import torch as _torch\n"
        "_orig_torch_load = _torch.load\n"
        "def _safe_torch_load(*args, **kwargs):\n"
        "    kwargs.setdefault('weights_only', False)\n"
        "    return _orig_torch_load(*args, **kwargs)\n"
        "_torch.load = _safe_torch_load\n"
        "# --- END_PY313_PT2_COMPAT_PATCH ---\n"
    )

    for py_file in REPO_DIR.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8", errors="ignore")
        orig = text
        if "PY313_PT2_COMPAT_PATCH" not in text:
            # Place after any __future__ imports
            lines = text.splitlines(keepends=True)
            insert_idx = 0
            for idx, line in enumerate(lines):
                if "__future__" in line:
                    insert_idx = idx + 1
            lines.insert(insert_idx, compat_header)
            text = "".join(lines)

        # Fix hardcoded cuda:0 in loss.py
        if py_file.name == "loss.py":
            text = text.replace("device='cuda:0'", "device=device")
            text = text.replace("gain[3] - 1", "(gain[3] - 1).clamp(min=0)")
            text = text.replace("gain[2] - 1", "(gain[2] - 1).clamp(min=0)")

        # Fix cache suffix in datasets.py to avoid Ultralytics cache collision
        if py_file.name == "datasets.py":
            text = text.replace(".with_suffix('.cache')", ".with_suffix('.yolov7.cache')")

        # Fix SummaryWriter in train.py
        if py_file.name == "train.py":
            text = text.replace(
                "from torch.utils.tensorboard import SummaryWriter",
                "class SummaryWriter:\n"
                "    def __init__(self, *a, **k): pass\n"
                "    def add_scalar(self, *a, **k): pass\n"
                "    def add_histogram(self, *a, **k): pass\n"
                "    def add_image(self, *a, **k): pass\n"
                "    def add_graph(self, *a, **k): pass\n"
                "    def close(self): pass"
            )

        # Fix plot_labels in plots.py if seaborn/pandas not full
        if py_file.name == "plots.py":
            text = text.replace(
                "def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):",
                "def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):\n    return"
            )

        if text != orig:
            py_file.write_text(text, encoding="utf-8")
            print(f"[PATCHED] {py_file.relative_to(REPO_DIR)}")

def test_forward_backward():
    sys.path.insert(0, str(SHIMS_DIR))
    sys.path.insert(0, str(REPO_DIR))
    import torch
    import yaml
    from models.yolo import Model
    from utils.loss import ComputeLossOTA, ComputeLoss
    from utils.torch_utils import intersect_dicts

    weights_path = SCRIPT_DIR / "yolov7-tiny.pt"
    cfg_path = REPO_DIR / "cfg" / "training" / "yolov7-tiny.yaml"
    hyp_path = REPO_DIR / "data" / "hyp.scratch.tiny.yaml"

    with open(hyp_path, "r") as f:
        hyp = yaml.safe_load(f)

    device = torch.device("cpu")
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    model = Model(str(cfg_path), ch=3, nc=1, anchors=hyp.get("anchors")).to(device)
    state_dict = ckpt["model"].float().state_dict()
    state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=["anchor"])
    model.load_state_dict(state_dict, strict=False)
    print(f"[OK] Loaded YOLOv7-tiny model: transferred {len(state_dict)}/{len(model.state_dict())} tensors")

    nl = model.model[-1].nl
    hyp["box"] *= 3.0 / nl
    hyp["cls"] *= 1 / 80.0 * 3.0 / nl
    hyp["obj"] *= (416 / 640) ** 2 * 3.0 / nl
    hyp["label_smoothing"] = 0.0
    model.nc = 1
    model.hyp = hyp
    model.gr = 1.0
    model.names = ["tree-top"]

    compute_loss_ota = ComputeLossOTA(model)
    compute_loss = ComputeLoss(model)

    # Dummy batch: 2 images of 416x416, 3 targets
    imgs = torch.rand(2, 3, 416, 416, device=device)
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.14, 0.14],
        [0, 0, 0.2, 0.3, 0.12, 0.15],
        [1, 0, 0.7, 0.6, 0.15, 0.13],
    ], dtype=torch.float32, device=device)

    model.train()
    pred = model(imgs)
    loss, loss_items = compute_loss_ota(pred, targets, imgs)
    loss.backward()
    print(f"[OK] Forward + ComputeLossOTA + Backward succeeded! Loss items: {loss_items.tolist()}")

if __name__ == "__main__":
    create_shims()
    patch_repo()
    test_forward_backward()
