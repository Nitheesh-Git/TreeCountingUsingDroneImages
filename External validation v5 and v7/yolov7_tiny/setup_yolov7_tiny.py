import os
import sys
import subprocess
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR / "yolov7_repo"
WEIGHTS_PATH = SCRIPT_DIR / "yolov7-tiny.pt"
WEIGHTS_URL = "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt"
GIT_URL = "https://github.com/WongKinYiu/yolov7.git"

def main():
    print("=== Step 1: Checking Python Environment Packages ===")
    for pkg in ["torch", "torchvision", "numpy", "scipy", "cv2", "yaml", "tqdm", "matplotlib", "pandas", "seaborn", "PIL", "tensorboard"]:
        try:
            m = __import__(pkg)
            print(f"  [OK] {pkg}: {getattr(m, '__version__', 'installed')}")
        except Exception as e:
            print(f"  [MISSING] {pkg}: {e}")

    print("\n=== Step 2: Cloning WongKinYiu/yolov7 (if needed) ===")
    if not REPO_DIR.exists():
         subprocess.run(["git", "clone", "--depth", "1", GIT_URL, str(REPO_DIR)], check=True)
         print(f"  [OK] Cloned into {REPO_DIR}")
    else:
        print(f"  [OK] Repo already exists at {REPO_DIR}")

    print("\n=== Step 3: Downloading yolov7-tiny.pt (if needed) ===")
    if not WEIGHTS_PATH.exists() or WEIGHTS_PATH.stat().st_size < 1_000_000:
        print(f"  Downloading {WEIGHTS_URL} -> {WEIGHTS_PATH} ...")
        urllib.request.urlretrieve(WEIGHTS_URL, str(WEIGHTS_PATH))
        print(f"  [OK] Downloaded ({WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"  [OK] Weights already exist ({WEIGHTS_PATH.stat().st_size / (1024*1024):.2f} MB)")

if __name__ == "__main__":
    main()
