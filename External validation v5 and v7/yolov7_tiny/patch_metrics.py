from pathlib import Path

p = Path(__file__).resolve().parent / "yolov7_repo" / "utils" / "metrics.py"
text = p.read_text(encoding="utf-8")
text = text.replace(
    "ap = np.trapz(np.interp(x, mrec, mpre), x)",
    "trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))\n        ap = trapz_fn(np.interp(x, mrec, mpre), x)"
)
p.write_text(text, encoding="utf-8")
print("[SUCCESS] metrics.py patched successfully for numpy 2.x trapezoid!")
