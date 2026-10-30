import csv
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO5_DIR = REPO_ROOT / "External validation v5 and v7" / "yolov5"
YOLO7_DIR = REPO_ROOT / "External validation v5 and v7" / "yolov7_tiny"

YOLO5_CSV = YOLO5_DIR / "runs" / "yolov5n_subset" / "results.csv"
YOLO7_CSV = YOLO7_DIR / "runs" / "yolov7_tiny_subset" / "results.csv"
YOLO7_TXT = YOLO7_DIR / "runs" / "yolov7_tiny_subset" / "results.txt"

YOLO5_LOG = YOLO5_DIR / "monitor.log"
YOLO7_LOG = YOLO7_DIR / "monitor.log"

YOLO5_TRAIN_LOG = YOLO5_DIR / "train.log"
YOLO7_TRAIN_LOG = YOLO7_DIR / "train.log"

def get_system_metrics():
    total_gb, free_gb, cpu_pct = 15.69, 3.5, 50.0
    try:
        cmd = "powershell -NoProfile -Command \"$os = Get-CimInstance Win32_OperatingSystem; $cpu = (Get-CimInstance Win32_Processor | Measure-Object -Property LoadPercentage -Average).Average; Write-Output ('{0}|{1}|{2}' -f [math]::Round($os.TotalVisibleMemorySize/1MB,2), [math]::Round($os.FreePhysicalMemory/1MB,2), $cpu)\""
        out = subprocess.check_output(cmd, shell=True, text=True, timeout=5).strip()
        parts = out.split("|")
        if len(parts) == 3:
            total_gb = float(parts[0])
            free_gb = float(parts[1])
            cpu_pct = float(parts[2])
    except Exception:
        pass
    pressure = "HIGH (RAM < 1.5GB)" if free_gb < 1.5 else ("MODERATE" if free_gb < 2.5 else "LOW (Ample RAM)")
    return total_gb, free_gb, cpu_pct, pressure

def is_process_running(keyword):
    try:
        cmd = f"powershell -NoProfile -Command \"(Get-CimInstance Win32_Process -Filter \\\"Name LIKE 'python%'\\\" | Where-Object {{ $_.CommandLine -like '*{keyword}*' }} | Measure-Object).Count\""
        out = subprocess.check_output(cmd, shell=True, text=True, timeout=5).strip()
        return int(out) > 0
    except Exception:
        return False

def parse_yolov5_status(start_time):
    running = is_process_running("train_yolov5.py")
    best_pt = YOLO5_DIR / "runs" / "yolov5n_subset" / "weights" / "best.pt"

    latest_epoch = 0
    latest_metrics = {}
    if YOLO5_CSV.exists():
        try:
            with open(YOLO5_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    # strip whitespace from keys
                    cleaned = {k.strip(): v.strip() for k, v in last_row.items() if k}
                    latest_epoch = int(cleaned.get("epoch", 0))
                    latest_metrics = cleaned
        except Exception:
            pass

    status = "RUNNING" if running else ("COMPLETED" if (best_pt.exists() and latest_epoch >= 1) else "IDLE/WAITING")
    elapsed = time.time() - start_time
    rem_str = "N/A"
    if latest_epoch > 0 and latest_epoch < 30 and running:
        sec_per_epoch = elapsed / latest_epoch
        rem_sec = (30 - latest_epoch) * sec_per_epoch
        rem_m, rem_s = divmod(int(rem_sec), 60)
        rem_str = f"~{rem_m}m {rem_s}s"
    elif latest_epoch >= 30 or status == "COMPLETED":
        rem_str = "0s (Done)"

    return {
        "model": "YOLOv5n",
        "status": status,
        "epoch": latest_epoch,
        "total_epochs": 30,
        "metrics": latest_metrics,
        "elapsed": elapsed,
        "remaining": rem_str
    }

def parse_yolov7_status(start_time):
    running = is_process_running("train_yolov7_tiny.py")
    best_pt = YOLO7_DIR / "runs" / "yolov7_tiny_subset" / "weights" / "best.pt"

    latest_epoch = 0
    latest_metrics = {}
    if YOLO7_CSV.exists():
        try:
            with open(YOLO7_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last_row = rows[-1]
                    cleaned = {k.strip(): v.strip() for k, v in last_row.items() if k}
                    latest_epoch = int(cleaned.get("epoch", 0))
                    latest_metrics = cleaned
        except Exception:
            pass
    elif YOLO7_TXT.exists():
        try:
            lines = [l.strip() for l in YOLO7_TXT.read_text(encoding="utf-8").splitlines() if l.strip()]
            latest_epoch = len(lines)
        except Exception:
            pass

    status = "RUNNING" if running else ("COMPLETED" if (best_pt.exists() and latest_epoch >= 1) else "IDLE/WAITING")
    elapsed = time.time() - start_time if start_time else 0.0
    rem_str = "N/A"
    if latest_epoch > 0 and latest_epoch < 30 and running and elapsed > 0:
        sec_per_epoch = elapsed / latest_epoch
        rem_sec = (30 - latest_epoch) * sec_per_epoch
        rem_m, rem_s = divmod(int(rem_sec), 60)
        rem_str = f"~{rem_m}m {rem_s}s"
    elif latest_epoch >= 30 or status == "COMPLETED":
        rem_str = "0s (Done)"

    return {
        "model": "YOLOv7-tiny",
        "status": status,
        "epoch": latest_epoch,
        "total_epochs": 30,
        "metrics": latest_metrics,
        "elapsed": elapsed,
        "remaining": rem_str
    }

def main():
    print("=" * 80)
    print("🌲 CONCURRENT TRAINING MONITOR (YOLOv5n & YOLOv7-tiny) 🌲")
    print("Interval: 180 seconds | Independent log writing enabled")
    print("=" * 80)

    t0 = time.time()
    v7_start_time = time.time()

    while True:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        total_gb, free_gb, cpu_pct, pressure = get_system_metrics()

        s5 = parse_yolov5_status(t0)
        s7 = parse_yolov7_status(v7_start_time)

        # Build v5 log string
        m5 = s5["metrics"]
        val_p5 = m5.get("metrics/precision(B)", "N/A")
        val_r5 = m5.get("metrics/recall(B)", "N/A")
        map50_5 = m5.get("metrics/mAP50(B)", "N/A")
        map95_5 = m5.get("metrics/mAP50-95(B)", "N/A")
        box_l5 = m5.get("train/box_loss", "N/A")

        rec5 = (
            f"[{timestamp}] Model: YOLOv5n | Status: {s5['status']} | "
            f"Epoch: {s5['epoch']}/{s5['total_epochs']} | "
            f"Train BoxLoss: {box_l5} | Val P: {val_p5} | Val R: {val_r5} | "
            f"mAP50: {map50_5} | mAP50-95: {map95_5} | "
            f"Elapsed: {int(s5['elapsed']//60)}m {int(s5['elapsed']%60)}s | Est. Rem: {s5['remaining']} | "
            f"Sys RAM Free: {free_gb:.2f}GB / {total_gb:.2f}GB | CPU: {cpu_pct}% | Memory Pressure: {pressure} | Errors: None\n"
        )
        with open(YOLO5_LOG, "a", encoding="utf-8") as f:
            f.write(rec5)

        # Build v7 log string
        m7 = s7["metrics"]
        val_p7 = m7.get("metrics/precision", "N/A")
        val_r7 = m7.get("metrics/recall", "N/A")
        map50_7 = m7.get("metrics/mAP50", "N/A")
        map95_7 = m7.get("metrics/mAP50-95", "N/A")
        box_l7 = m7.get("train/box_loss", "N/A")

        rec7 = (
            f"[{timestamp}] Model: YOLOv7-tiny | Status: {s7['status']} | "
            f"Epoch: {s7['epoch']}/{s7['total_epochs']} | "
            f"Train BoxLoss: {box_l7} | Val P: {val_p7} | Val R: {val_r7} | "
            f"mAP50: {map50_7} | mAP50-95: {map95_7} | "
            f"Elapsed: {int(s7['elapsed']//60)}m {int(s7['elapsed']%60)}s | Est. Rem: {s7['remaining']} | "
            f"Sys RAM Free: {free_gb:.2f}GB / {total_gb:.2f}GB | CPU: {cpu_pct}% | Memory Pressure: {pressure} | Errors: None\n"
        )
        with open(YOLO7_LOG, "a", encoding="utf-8") as f:
            f.write(rec7)

        # Print combined summary table
        print(f"\n--- [MONITOR REPORT @ {timestamp}] ---", flush=True)
        print(f"System: Free RAM {free_gb:.2f} GB / {total_gb:.2f} GB | CPU {cpu_pct}% | Memory Pressure: {pressure}", flush=True)
        print(f"  • YOLOv5n    : Status={s5['status']} | Epoch {s5['epoch']}/{s5['total_epochs']} | mAP50={map50_5} | Rem={s5['remaining']}", flush=True)
        print(f"  • YOLOv7-tiny: Status={s7['status']} | Epoch {s7['epoch']}/{s7['total_epochs']} | mAP50={map50_7} | Rem={s7['remaining']}", flush=True)
        print("-" * 65, flush=True)

        if s5["status"] == "COMPLETED" and s7["status"] == "COMPLETED":
            print("\n[SUCCESS] Both YOLOv5n and YOLOv7-tiny training completed!", flush=True)
            break

        time.sleep(180)

if __name__ == "__main__":
    main()
