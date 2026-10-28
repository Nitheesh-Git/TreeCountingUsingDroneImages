"""
Build the reproducible half-size training subset used by the comparison models.

Selection protocol
------------------
* The Roboflow export augmented every training source image into 3 versions
  (brightness + salt-and-pepper noise). Files sharing the stem before ".rf."
  are versions of the same source scene, so selection is done per group,
  never per file. Sampling single files would keep ~87% of the unique scenes
  and would not be a real "half" of the data.
* Groups whose stem also appears in the valid or test split are excluded as a
  precaution against near-duplicate leakage (the original splits are kept
  unchanged; this only affects which training images are eligible).
* Groups are stratified by NEON site code (ABBY, BART, ...) and shuffled with
  a fixed seed; each site contributes groups until it is as close as possible
  to half of that site's original image count. Because whole groups (3 or 6
  files) undershoot per site, a seeded top-up pass over the remaining groups
  then brings the total as close as possible to fraction * 1191.

Outputs (committed, small)
--------------------------
* subset_manifest.txt  - selected image paths relative to the repo root
* subset_stats.json    - seed, counts, label validation and distribution stats
* subset_stats.md      - the same statistics as a readable table

The original dataset and data.yaml are only read, never modified.

Usage:
    python comparison_models/common/make_subset.py [--seed 0] [--fraction 0.5]
"""

import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = REPO_ROOT / "datasets" / "Tree-Top-View"
OUT_DIR = Path(__file__).resolve().parent
IMG_SIZE = 416  # all images in the export are 416x416 (verified in validate step)
SITE_RE = re.compile(r"^(?:\d{4}_)?([A-Z]+)")


def group_of(path):
    return path.name.split(".rf.")[0]


def site_of(path):
    return SITE_RE.match(path.name).group(1)


def label_path(img_path):
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def read_labels(img_path):
    """Return (boxes, problems) for one image; boxes are (cls, x, y, w, h)."""
    lbl = label_path(img_path)
    if not lbl.exists():
        return [], ["missing label file"]
    boxes, problems = [], []
    for i, line in enumerate(lbl.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"line {i}: expected 5 values, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            problems.append(f"line {i}: non-numeric value")
            continue
        if cls != 0:
            problems.append(f"line {i}: class {cls} (dataset has nc=1)")
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)) or w <= 0 or h <= 0:
            problems.append(f"line {i}: coordinates out of range")
        boxes.append((cls, x, y, w, h))
    return boxes, problems


def describe(images):
    """Instance and box-size statistics for a list of image paths."""
    per_image, widths, heights, areas = [], [], [], []
    size_buckets = Counter()
    for p in images:
        boxes, _ = read_labels(p)
        per_image.append(len(boxes))
        for _, _, _, w, h in boxes:
            wp, hp = w * IMG_SIZE, h * IMG_SIZE
            widths.append(wp)
            heights.append(hp)
            areas.append(wp * hp)
            # COCO size convention, measured at the 416px training resolution
            size_buckets["small (<32^2 px)" if wp * hp < 32 ** 2
                         else "medium (32^2-96^2 px)" if wp * hp < 96 ** 2
                         else "large (>=96^2 px)"] += 1

    def q(values, frac):
        s = sorted(values)
        return round(s[min(len(s) - 1, int(frac * len(s)))], 2) if s else None

    n_boxes = len(areas)
    return {
        "images": len(images),
        "source_groups": len({group_of(p) for p in images}),
        "instances": n_boxes,
        "images_without_trees": sum(1 for c in per_image if c == 0),
        "instances_per_image": {
            "mean": round(statistics.mean(per_image), 2),
            "median": statistics.median(per_image),
            "p90": q(per_image, 0.9),
            "max": max(per_image),
        },
        "box_width_px": {"median": q(widths, 0.5), "p10": q(widths, 0.1), "p90": q(widths, 0.9)},
        "box_height_px": {"median": q(heights, 0.5), "p10": q(heights, 0.1), "p90": q(heights, 0.9)},
        "box_area_px": {"median": q(areas, 0.5), "p10": q(areas, 0.1), "p90": q(areas, 0.9)},
        "box_size_share": {k: round(v / n_boxes, 4) for k, v in sorted(size_buckets.items())},
        "sites": dict(sorted(Counter(site_of(p) for p in images).items())),
    }


def validate_split(split):
    from PIL import Image  # only needed here

    images = sorted((DATASET_DIR / split / "images").glob("*.jpg"))
    labels = sorted((DATASET_DIR / split / "labels").glob("*.txt"))
    problems = {}
    sizes = Counter()
    for p in images:
        _, issues = read_labels(p)
        if issues:
            problems[p.name] = issues
        with Image.open(p) as im:
            sizes[f"{im.size[0]}x{im.size[1]}"] += 1
    orphan_labels = {l.stem for l in labels} - {p.stem for p in images}
    return images, {
        "images": len(images),
        "label_files": len(labels),
        "orphan_label_files": len(orphan_labels),
        "images_with_label_problems": len(problems),
        "problem_examples": dict(list(problems.items())[:10]),
        "image_sizes": dict(sizes),
    }


def select(train_images, excluded_groups, fraction, seed):
    by_site = defaultdict(lambda: defaultdict(list))
    for p in train_images:
        g = group_of(p)
        if g not in excluded_groups:
            by_site[site_of(p)][g].append(p)

    site_totals = Counter(site_of(p) for p in train_images)
    rng = random.Random(seed)
    selected, leftovers = [], []
    for site in sorted(by_site):
        groups = sorted(by_site[site])  # sort first so the shuffle is deterministic
        rng.shuffle(groups)
        target = site_totals[site] * fraction
        taken = 0
        for i, g in enumerate(groups):
            size = len(by_site[site][g])
            if taken and abs(taken + size - target) >= abs(taken - target):
                leftovers.extend(by_site[site][h] for h in groups[i:])
                break
            selected.extend(by_site[site][g])
            taken += size

    # Per-site rounding to whole groups undershoots; top up from the remaining
    # groups (seeded order) until the total is closest to the global target.
    target = len(train_images) * fraction
    rng.shuffle(leftovers)
    for files in leftovers:
        if abs(len(selected) + len(files) - target) >= abs(len(selected) - target):
            continue
        selected.extend(files)
    return sorted(selected)


def write_markdown(stats, path):
    full, sub = stats["full_train"], stats["subset_train"]
    rows = [
        ("Images", full["images"], sub["images"]),
        ("Source scene groups", full["source_groups"], sub["source_groups"]),
        ("Tree instances", full["instances"], sub["instances"]),
        ("Images without trees", full["images_without_trees"], sub["images_without_trees"]),
        ("Instances / image (mean)", full["instances_per_image"]["mean"], sub["instances_per_image"]["mean"]),
        ("Instances / image (median)", full["instances_per_image"]["median"], sub["instances_per_image"]["median"]),
        ("Instances / image (p90)", full["instances_per_image"]["p90"], sub["instances_per_image"]["p90"]),
        ("Instances / image (max)", full["instances_per_image"]["max"], sub["instances_per_image"]["max"]),
        ("Box width px (median)", full["box_width_px"]["median"], sub["box_width_px"]["median"]),
        ("Box height px (median)", full["box_height_px"]["median"], sub["box_height_px"]["median"]),
        ("Box area px (p10 / median / p90)",
         f'{full["box_area_px"]["p10"]} / {full["box_area_px"]["median"]} / {full["box_area_px"]["p90"]}',
         f'{sub["box_area_px"]["p10"]} / {sub["box_area_px"]["median"]} / {sub["box_area_px"]["p90"]}'),
    ]
    for k in full["box_size_share"]:
        rows.append((f"Share of boxes: {k}", full["box_size_share"][k], sub["box_size_share"].get(k, 0)))

    lines = [
        "# Half-size training subset: statistics",
        "",
        f"Generated by `comparison_models/common/make_subset.py` with seed **{stats['seed']}** "
        f"and fraction **{stats['fraction']}**.",
        "",
        "| Statistic | Full train split | Subset |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in rows],
        "",
        "## Images per site",
        "",
        "| Site | Full train | Subset |",
        "|---|---|---|",
        *[f"| {s} | {n} | {sub['sites'].get(s, 0)} |" for s, n in full["sites"].items()],
        "",
        "## Excluded groups (stem also present in valid/test)",
        "",
        *[f"- `{g}`" for g in stats["excluded_groups"]],
        "",
        "## Label validation (original splits, read-only)",
        "",
        "| Split | Images | Label files | Orphan labels | Images with label problems | Image sizes |",
        "|---|---|---|---|---|---|",
        *[f"| {s} | {v['images']} | {v['label_files']} | {v['orphan_label_files']} | "
          f"{v['images_with_label_problems']} | {v['image_sizes']} |"
          for s, v in stats["validation"].items()],
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Create the reproducible half-size training subset.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for group selection")
    parser.add_argument("--fraction", type=float, default=0.5, help="Target fraction of the train split")
    args = parser.parse_args()

    validation, split_images = {}, {}
    for split in ("train", "valid", "test"):
        split_images[split], validation[split] = validate_split(split)

    held_out_groups = {group_of(p) for s in ("valid", "test") for p in split_images[s]}
    excluded = sorted({group_of(p) for p in split_images["train"]} & held_out_groups)

    subset = select(split_images["train"], set(excluded), args.fraction, args.seed)

    # Safety checks: no held-out file or group can be in the subset
    held_out_names = {p.name for s in ("valid", "test") for p in split_images[s]}
    assert not held_out_names & {p.name for p in subset}
    assert not held_out_groups & {group_of(p) for p in subset}
    assert all(label_path(p).exists() for p in subset)

    manifest = OUT_DIR / "subset_manifest.txt"
    manifest.write_text("".join(f"{p.relative_to(REPO_ROOT).as_posix()}\n" for p in subset))

    stats = {
        "seed": args.seed,
        "fraction": args.fraction,
        "method": "group-level (Roboflow augmentation triplets kept together), "
                  "stratified by site, held-out-overlap groups excluded",
        "excluded_groups": excluded,
        "validation": validation,
        "full_train": describe(split_images["train"]),
        "subset_train": describe(subset),
        "valid": describe(split_images["valid"]),
        "test": describe(split_images["test"]),
    }
    (OUT_DIR / "subset_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    write_markdown(stats, OUT_DIR / "subset_stats.md")

    print(f"Selected {len(subset)} / {len(split_images['train'])} train images "
          f"({stats['subset_train']['source_groups']} source groups, seed={args.seed})")
    print(f"Excluded {len(excluded)} groups that overlap valid/test by name")
    print(f"Manifest : {manifest.relative_to(REPO_ROOT)}")
    print(f"Stats    : comparison_models/common/subset_stats.md")


if __name__ == "__main__":
    main()
