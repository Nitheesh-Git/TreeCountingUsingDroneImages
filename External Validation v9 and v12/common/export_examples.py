"""
Export example images for the report into examples/: raw training, raw test,
ground-truth-labelled and model-prediction images (3 of each, seed 0).

Usage:
    python "External Validation v9 and v12/common/export_examples.py"
"""

import random
import re
import shutil
from pathlib import Path

import cv2

PROJECT = Path(__file__).resolve().parents[1]  # "External Validation v9 and v12/"
REPO = PROJECT.parent
OUT = PROJECT / "examples"
DS = REPO / "datasets" / "Tree-Top-View"
SITE = re.compile(r"^(?:\d{4}_)?([A-Z]+)")
WEIGHTS = {
    "YOLOv9t": PROJECT / "yolov9/runs/yolov9t_half_subset/weights/best.pt",
    "YOLOv12n": PROJECT / "yolov12/runs/yolov12n_half_subset/weights/best.pt",
}
SCALE = 2  # draw on a 2x upscaled copy (832 px) so boxes and text are readable in a report


def labels(img):
    lbl = img.parent.parent / "labels" / f"{img.stem}.txt"
    return [list(map(float, l.split()[1:5])) for l in lbl.read_text().splitlines() if l.strip()]


def pick(images, seed):
    """One sparse (2-5), one medium (8-12), one dense (16+) image, from three different sites."""
    rng = random.Random(seed)
    chosen, used_sites = [], set()
    for lo, hi in [(2, 5), (8, 12), (16, 99)]:
        cands = [p for p in images if lo <= len(labels(p)) <= hi and SITE.match(p.name).group(1) not in used_sites]
        p = rng.choice(sorted(cands))
        chosen.append(p)
        used_sites.add(SITE.match(p.name).group(1))
    return chosen


def short(p, i, kind):
    return f"{i}_{kind}_{SITE.match(p.name).group(1)}_{len(labels(p))}trees"


def draw(img_path, boxes, color, title):
    im = cv2.imread(str(img_path))
    im = cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_CUBIC)
    h, w = im.shape[:2]
    for x, y, bw, bh in boxes:
        cv2.rectangle(im, (int((x - bw / 2) * w), int((y - bh / 2) * h)),
                      (int((x + bw / 2) * w), int((y + bh / 2) * h)), color, 2)
    cv2.rectangle(im, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.putText(im, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return im


def main():
    if OUT.exists():
        shutil.rmtree(OUT)  # only ever removes the folder this script creates
    for d in ("1_training_set", "2_test_dataset", "3_labelled_ground_truth", "4_model_predictions"):
        (OUT / d).mkdir(parents=True)

    train = [REPO / l for l in (PROJECT / "common/subset_manifest.txt").read_text().split()]
    test = sorted((DS / "test" / "images").glob("*.jpg"))
    tr, te = pick(train, 0), pick(test, 0)

    for i, p in enumerate(tr, 1):
        shutil.copy2(p, OUT / "1_training_set" / f"{short(p, i, 'train')}.jpg")
        cv2.imwrite(str(OUT / "3_labelled_ground_truth" / f"{short(p, i, 'train_labelled')}.jpg"),
                    draw(p, labels(p), (0, 255, 0), f"Ground truth: {len(labels(p))} trees"))
    for i, p in enumerate(te, 1):
        shutil.copy2(p, OUT / "2_test_dataset" / f"{short(p, i, 'test')}.jpg")

    from ultralytics import YOLO
    for i, p in enumerate(te, 1):
        gt = labels(p)
        cv2.imwrite(str(OUT / "4_model_predictions" / f"{short(p, i, 'test')}_0_ground_truth.jpg"),
                    draw(p, gt, (0, 255, 0), f"Ground truth: {len(gt)} trees"))
        for name, w in WEIGHTS.items():
            r = YOLO(str(w)).predict(str(p), conf=0.25, imgsz=416, verbose=False)[0]
            boxes = [list(map(float, b)) for b in r.boxes.xywhn.tolist()]
            cv2.imwrite(str(OUT / "4_model_predictions" / f"{short(p, i, 'test')}_{name}.jpg"),
                        draw(p, boxes, (0, 165, 255), f"{name}: {len(boxes)} trees (GT {len(gt)}), conf 0.25"))

    (OUT / "README.txt").write_text(
        "Example images - Tree Counting Using Drone Images (Tree-Top-View, 416x416 NEON aerial tiles)\n\n"
        "1_training_set/          3 raw images from the 594-image training subset used for YOLOv9t/YOLOv12n\n"
        "2_test_dataset/          3 raw images from the unseen 55-image test split\n"
        "3_labelled_ground_truth/ the 3 training images with ground-truth tree boxes (green), shown at 2x\n"
        "4_model_predictions/     the 3 test images: ground truth (green) vs YOLOv9t and YOLOv12n\n"
        "                         detections (orange, conf 0.25), shown at 2x\n\n"
        "Images were chosen with a fixed seed (0): one sparse, one medium and one dense image,\n"
        "each from a different NEON site. File names give the site and the ground-truth tree count.\n"
        "Regenerate with: python \"External Validation v9 and v12/common/export_examples.py\"\n")
    for f in sorted(OUT.rglob("*.jpg")):
        print(f.relative_to(OUT))


if __name__ == "__main__":
    main()
