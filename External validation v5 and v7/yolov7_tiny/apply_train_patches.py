import re
from pathlib import Path

train_path = Path(__file__).resolve().parent / "yolov7_repo" / "train.py"
text = train_path.read_text(encoding="utf-8")

# 1. Update init_seeds to use opt.seed
text = re.sub(
    r"init_seeds\(2 \+ rank\)",
    "init_seeds(getattr(opt, 'seed', 42) + rank)",
    text
)

# 2. Add patience and seed arguments to parser
if "--patience" not in text:
    text = re.sub(
        r"(\s+parser\.add_argument\('--v5-metric'[^\n]+\n)",
        r"\1    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')\n    parser.add_argument('--seed', type=int, default=42, help='Random seed')\n",
        text
    )

# 3. Add results.csv logging
if "results.csv" not in text:
    target_write = "with open(results_file, 'a') as f:\n                f.write(s + '%10.4g' * 7 % results + '\\n')"
    csv_code = (
        "with open(results_file, 'a') as f:\n"
        "                f.write(s + '%10.4g' * 7 % results + '\\n')\n"
        "            csv_file = save_dir / 'results.csv'\n"
        "            if not csv_file.exists():\n"
        "                with open(csv_file, 'w', encoding='utf-8') as f_csv:\n"
        "                    f_csv.write('epoch,train/box_loss,train/obj_loss,train/cls_loss,metrics/precision,metrics/recall,metrics/mAP50,metrics/mAP50-95,val/box_loss,val/obj_loss,val/cls_loss\\n')\n"
        "            with open(csv_file, 'a', encoding='utf-8') as f_csv:\n"
        "                f_csv.write(f'{epoch+1},{mloss[0]:.5f},{mloss[1]:.5f},{mloss[2]:.5f},{results[0]:.5f},{results[1]:.5f},{results[2]:.5f},{results[3]:.5f},{results[4]:.5f},{results[5]:.5f},{results[6]:.5f}\\n')"
    )
    text = text.replace(target_write, csv_code)

train_path.write_text(text, encoding="utf-8")
print("[SUCCESS] train.py successfully patched!")
