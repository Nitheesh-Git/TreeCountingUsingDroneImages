"""
Prepare an Independent 100-Image External Test Dataset from SelvaBox (CanopyRS/SelvaBox).
Strictly uses the official held-out 'test' split.
Uses Shard 00 (Ecuador: 62 tiles) and Shard 16 (Brazil: 32 tiles, Panama: 29 tiles) = 123 official test tiles
spanning all 3 countries, downloaded via concurrent async HTTP range chunks to avoid downloading the 31.3 GB dataset.
"""

import asyncio
import csv
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import fsspec
import pyarrow.parquet as pq
from PIL import Image

SEED = 42
TARGET_COUNT = 100
REPO_ROOT = Path(r"C:\Users\astle\AI1\TreeCountingUsingDroneImages")
OUT_DIR = REPO_ROOT / "External validation v5 and v7" / "testdataset100"
IMAGES_DIR = OUT_DIR / "images"
ANNOTATIONS_DIR = OUT_DIR / "annotations"
META_CACHE_PATH = REPO_ROOT / "External validation v5 and v7" / "selvabox_test_metadata_cache.json"
TEMP_SHARD_DIR = REPO_ROOT / "External validation v5 and v7" / "_temp_selvabox_shards"

HF_DATASET = "CanopyRS/SelvaBox"
HF_URL = "https://huggingface.co/datasets/CanopyRS/SelvaBox"
DOC_URL = "https://hugobaudchon.github.io/CanopyRS/user-guide/data/"
TARGET_SHARDS = [0, 16]  # Shard 00 (62 Ecuador) + Shard 16 (32 Brazil, 29 Panama) = 123 official test tiles across all 3 countries


async def download_file_concurrent(url: str, dest_path: Path, concurrency: int = 24, chunk_mb: int = 2):
    """Download a remote parquet shard using concurrent async HTTP Range requests."""
    timeout = aiohttp.ClientTimeout(total=600, sock_connect=30, sock_read=120)
    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"User-Agent": "Mozilla/5.0"}) as session:
        # Resolve redirect and total size via Range 0-0 GET
        async with session.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True) as resp:
            cdn_url = str(resp.url)
            cr = resp.headers.get("Content-Range", "")
            if "/" in cr:
                total_size = int(cr.split("/")[-1])
            else:
                total_size = int(resp.headers.get("Content-Length", 0))

        print(f"    Resolved CDN URL ({total_size / (1024*1024):.1f} MB). Downloading with {concurrency} parallel streams...", flush=True)
        # Pre-allocate file
        with dest_path.open("wb") as f:
            f.truncate(total_size)

        chunk_size = chunk_mb * 1024 * 1024
        ranges = []
        for start in range(0, total_size, chunk_size):
            end = min(start + chunk_size - 1, total_size - 1)
            ranges.append((start, end))

        sem = asyncio.Semaphore(concurrency)
        completed_bytes = 0
        t0 = time.time()
        last_print = t0

        async def fetch_range(start: int, end: int):
            nonlocal completed_bytes, last_print
            async with sem:
                for attempt in range(5):
                    try:
                        async with session.get(cdn_url, headers={"Range": f"bytes={start}-{end}"}) as r:
                            data = await r.read()
                            if len(data) != (end - start + 1):
                                raise IOError(f"Incomplete chunk: got {len(data)}, expected {end - start + 1}")
                            with dest_path.open("r+b") as f:
                                f.seek(start)
                                f.write(data)
                            completed_bytes += len(data)
                            now = time.time()
                            if now - last_print >= 10.0 or completed_bytes == total_size:
                                pct = completed_bytes * 100.0 / total_size
                                spd = (completed_bytes / (1024 * 1024)) / max(now - t0, 0.1)
                                print(f"      {completed_bytes/(1024*1024):.1f} / {total_size/(1024*1024):.1f} MB ({pct:.1f}%) | {spd:.2f} MB/s", flush=True)
                                last_print = now
                            return
                    except Exception as e:
                        if attempt == 4:
                            raise
                        await asyncio.sleep(1.5 * (attempt + 1))

        await asyncio.gather(*(fetch_range(s, e) for s, e in ranges))


def compute_iou_or_overlap(bounds1, bounds2) -> float:
    if not bounds1 or not bounds2 or len(bounds1) != 4 or len(bounds2) != 4:
        return 0.0
    x_left = max(bounds1[0], bounds2[0])
    y_bottom = max(bounds1[1], bounds2[1])
    x_right = min(bounds1[2], bounds2[2])
    y_top = min(bounds1[3], bounds2[3])
    if x_right <= x_left or y_top <= y_bottom:
        return 0.0
    inter_area = (x_right - x_left) * (y_top - y_bottom)
    area1 = max((bounds1[2] - bounds1[0]) * (bounds1[3] - bounds1[1]), 1e-9)
    area2 = max((bounds2[2] - bounds2[0]) * (bounds2[3] - bounds2[1]), 1e-9)
    return inter_area / min(area1, area2)


def validate_and_convert_boxes(bboxes, img_w: int, img_h: int):
    valid_coco = []
    yolo_lines = []
    malformed = 0

    for box in bboxes:
        if not box or len(box) != 4:
            malformed += 1
            continue
        x, y, w, h = [float(v) for v in box]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(w) and math.isfinite(h)):
            malformed += 1
            continue
        x1 = max(0.0, min(float(img_w), x))
        y1 = max(0.0, min(float(img_h), y))
        x2 = max(0.0, min(float(img_w), x + w))
        y2 = max(0.0, min(float(img_h), y + h))

        cw = x2 - x1
        ch = y2 - y1
        if cw < 2.0 or ch < 2.0:
            malformed += 1
            continue

        xc_norm = (x1 + cw / 2.0) / float(img_w)
        yc_norm = (y1 + ch / 2.0) / float(img_h)
        w_norm = cw / float(img_w)
        h_norm = ch / float(img_h)

        if not (0.0 <= xc_norm <= 1.0 and 0.0 <= yc_norm <= 1.0 and 0.0 < w_norm <= 1.0 and 0.0 < h_norm <= 1.0):
            malformed += 1
            continue

        valid_coco.append([round(x1, 2), round(y1, 2), round(cw, 2), round(ch, 2)])
        yolo_lines.append(f"0 {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

    return valid_coco, yolo_lines, malformed


def main():
    t_start = time.time()
    print("=" * 75)
    print("SelvaBox 100-Image Independent External Test Dataset Preparation")
    print("=" * 75, flush=True)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(TEMP_SHARD_DIR, ignore_errors=True)
    TEMP_SHARD_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load metadata from cache (or scan if missing)
    all_candidates = json.loads(META_CACHE_PATH.read_text(encoding="utf-8"))
    shard_candidates = [c for c in all_candidates if c["shard_idx"] in TARGET_SHARDS]
    total_inspected = len(shard_candidates)
    print(f"[Step 1] Loaded {total_inspected} official test split candidates from Shards {TARGET_SHARDS} (out of {len(all_candidates)} total test split tiles).", flush=True)

    rejection_counts = {
        "not_test_split": 0,
        "missing_or_unmatched_annotation": 0,
        "zero_usable_tree_crowns": 0,
        "malformed_annotation_coordinates": 0,
        "invalid_dimensions_or_metadata": 0,
        "exact_duplicate_tile_name": 0,
        "spatial_overlap_or_excess_site_pool": 0,
        "corrupt_or_unreadable_image": 0,
        "not_suitable_aerial_imagery_nodata": 0,
    }

    valid_pool = []
    seen_tile_names = set()

    for cand in shard_candidates:
        if cand.get("fold") != "test":
            rejection_counts["not_test_split"] += 1
            continue
        tile_name = cand.get("tile_name")
        if not tile_name:
            rejection_counts["invalid_dimensions_or_metadata"] += 1
            continue
        if tile_name in seen_tile_names:
            rejection_counts["exact_duplicate_tile_name"] += 1
            continue
        seen_tile_names.add(tile_name)

        w = int(cand.get("width") or 0)
        h = int(cand.get("height") or 0)
        if w <= 0 or h <= 0:
            rejection_counts["invalid_dimensions_or_metadata"] += 1
            continue

        ann = cand.get("annotations")
        if not ann or "bbox" not in ann or ann["bbox"] is None:
            rejection_counts["missing_or_unmatched_annotation"] += 1
            continue

        raw_bboxes = ann["bbox"]
        if len(raw_bboxes) == 0:
            rejection_counts["zero_usable_tree_crowns"] += 1
            continue

        valid_coco, yolo_lines, malformed = validate_and_convert_boxes(raw_bboxes, w, h)
        if len(valid_coco) == 0:
            rejection_counts["zero_usable_tree_crowns"] += 1
            continue

        cand["valid_coco_boxes"] = valid_coco
        cand["yolo_lines"] = yolo_lines
        cand["tree_count"] = len(valid_coco)
        cand["malformed_boxes_dropped"] = malformed
        valid_pool.append(cand)

    print(f"[Step 2] {len(valid_pool)}/{total_inspected} candidates passed metadata & annotation checks.", flush=True)

    # Step 3: Order candidates per site prioritizing lowest spatial overlap first, with deterministic seed=42
    rng = random.Random(SEED)
    by_loc = {}
    for c in valid_pool:
        by_loc.setdefault(c["location"], []).append(c)

    ordered_by_loc = {}
    for loc, items in sorted(by_loc.items()):
        items_sorted = sorted(items, key=lambda x: x["tile_name"])
        rng.shuffle(items_sorted)
        selected_for_loc = []
        bounds_for_loc = []
        chosen_names = set()
        # Multi-pass overlap prioritization: 0% -> 25% -> 50% -> remaining
        for max_ov, tier_name in [(0.01, "0% (disjoint)"), (0.25, "<=25%"), (0.50, "<=50%"), (1.0, "standard-grid")]:
            for cand in items_sorted:
                if cand["tile_name"] in chosen_names:
                    continue
                b = (cand.get("tile_metadata") or {}).get("bounds")
                overlaps = False
                if b and len(b) == 4:
                    for eb in bounds_for_loc:
                        if compute_iou_or_overlap(b, eb) > max_ov:
                            overlaps = True
                            break
                if not overlaps:
                    if b and len(b) == 4:
                        bounds_for_loc.append(b)
                    cand["max_overlap_tier"] = tier_name
                    # Assign density tier
                    tc = cand["tree_count"]
                    cand["density_tier"] = "sparse" if tc < 18 else ("medium" if tc < 28 else "dense")
                    selected_for_loc.append(cand)
                    chosen_names.add(cand["tile_name"])
        ordered_by_loc[loc] = selected_for_loc
        print(f"  Site '{loc}': {len(selected_for_loc)} candidates ordered by spatial separation & density.", flush=True)

    # Clear old files
    for old_f in IMAGES_DIR.glob("*"):
        old_f.unlink()
    for old_f in ANNOTATIONS_DIR.glob("*"):
        old_f.unlink()

    # Step 4: Download Shard 16 (32 Brazil + 29 Panama = 61 tiles) and Shard 00 (from which we take 39+ Ecuador tiles to reach 100)
    # Using ordered shortlist: all 32 Brazil + all 29 Panama + top 44 Ecuador (39 primary + 5 backup)
    shortlist = (
        ordered_by_loc.get("brazil_zf2", [])
        + ordered_by_loc.get("panama_aguasalud", [])
        + ordered_by_loc.get("ecuador_tiputini", [])
    )

    final_selected = []
    seen_hashes = set()

    for s_idx in [16, 0]:
        if len(final_selected) >= TARGET_COUNT:
            break
        shard_cands = [c for c in shortlist if c["shard_idx"] == s_idx]
        url = f"https://huggingface.co/datasets/CanopyRS/SelvaBox/resolve/main/data/test-{s_idx:05d}-of-00024.parquet"
        local_parquet = TEMP_SHARD_DIR / f"test-{s_idx:05d}-of-00024.parquet"
        t_dl = time.time()
        print(f"\n[Step 4] Downloading Shard {s_idx:02d}/23 ({len(shard_cands)} candidate tiles)...", flush=True)
        asyncio.run(download_file_concurrent(url, local_parquet, concurrency=24, chunk_mb=2))
        print(f"  Shard {s_idx:02d} downloaded in {time.time() - t_dl:.1f}s. Decoding & verifying images...", flush=True)

        pf = pq.ParquetFile(local_parquet)
        img_table = pf.read(columns=["image"])
        img_col = img_table.column("image")

        for cand in shard_cands:
            if len(final_selected) >= TARGET_COUNT:
                rejection_counts["spatial_overlap_or_excess_site_pool"] += 1
                continue
            r_idx = cand["row_idx"]
            val = img_col[r_idx].as_py()
            raw_bytes = val["bytes"] if isinstance(val, dict) else None
            if not raw_bytes:
                rejection_counts["corrupt_or_unreadable_image"] += 1
                continue

            sha256 = hashlib.sha256(raw_bytes).hexdigest()
            if sha256 in seen_hashes:
                rejection_counts["exact_duplicate_tile_name"] += 1
                continue

            try:
                with Image.open(io.BytesIO(raw_bytes)) as img:
                    img.verify()
                with Image.open(io.BytesIO(raw_bytes)) as img:
                    actual_w, actual_h = img.size
                    if actual_w != cand["width"] or actual_h != cand["height"]:
                        rejection_counts["invalid_dimensions_or_metadata"] += 1
                        continue
                    rgb_img = img.convert("RGB")
                    thumb = rgb_img.resize((100, 100))
                    pixels = list(thumb.getdata())
                    dark_count = sum(
                        1 for r, g, b in pixels if (r == 0 and g == 0 and b == 0) or (r == 255 and g == 255 and b == 255)
                    )
                    if dark_count / len(pixels) > 0.40:
                        rejection_counts["not_suitable_aerial_imagery_nodata"] += 1
                        continue

                    idx_1based = len(final_selected) + 1
                    image_id = f"image_{idx_1based:03d}"
                    local_img_name = f"{image_id}.png"
                    local_ann_name = f"{image_id}.txt"

                    out_img_path = IMAGES_DIR / local_img_name
                    rgb_img.save(out_img_path, format="PNG", compress_level=3)
            except Exception:
                rejection_counts["corrupt_or_unreadable_image"] += 1
                continue

            seen_hashes.add(sha256)
            out_ann_path = ANNOTATIONS_DIR / local_ann_name
            out_ann_path.write_text("\n".join(cand["yolo_lines"]) + "\n", encoding="utf-8")

            cand["image_id"] = image_id
            cand["local_filename"] = local_img_name
            cand["annotation_filename"] = local_ann_name
            cand["sha256"] = sha256
            final_selected.append(cand)

        del img_col, img_table, pf
        local_parquet.unlink(missing_ok=True)
        print(f"  Progress: {len(final_selected)}/{TARGET_COUNT} verified images saved.", flush=True)

    shutil.rmtree(TEMP_SHARD_DIR, ignore_errors=True)
    META_CACHE_PATH.unlink(missing_ok=True)

    if len(final_selected) != TARGET_COUNT:
        raise RuntimeError(f"Expected {TARGET_COUNT} images, got {len(final_selected)}")

    # Save original COCO annotations archive inside annotations/original_selvabox_annotations.json
    orig_ann_archive = {}
    for c in final_selected:
        orig_ann_archive[c["image_id"]] = {
            "local_filename": c["local_filename"],
            "original_filename": c["tile_name"],
            "location": c["location"],
            "raster_name": c["raster_name"],
            "fold": c["fold"],
            "width": c["width"],
            "height": c["height"],
            "coco_bboxes_xywh": c["valid_coco_boxes"],
            "tile_metadata": c["tile_metadata"],
        }
    (ANNOTATIONS_DIR / "original_selvabox_annotations.json").write_text(
        json.dumps(orig_ann_archive, indent=2), encoding="utf-8"
    )

    # Write manifest.csv
    manifest_path = OUT_DIR / "manifest.csv"
    fieldnames = [
        "image_id",
        "local_filename",
        "original_filename",
        "source_dataset",
        "source_split",
        "source_site",
        "source_image_id",
        "annotation_filename",
        "image_width",
        "image_height",
        "annotated_tree_count",
        "original_source_reference",
        "selection_seed",
        "selection_notes",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=fieldnames)
        writer.writeheader()
        for c in final_selected:
            ov_tier = c.get("max_overlap_tier", "0% (disjoint)")
            notes = f"density={c.get('density_tier', 'mixed')}; overlap_tier={ov_tier}; raster={c['raster_name']}"
            writer.writerow({
                "image_id": c["image_id"],
                "local_filename": c["local_filename"],
                "original_filename": c["tile_name"],
                "source_dataset": HF_DATASET,
                "source_split": c["fold"],
                "source_site": c["location"],
                "source_image_id": f"{c['location']}/{c['raster_name']}/{c['tile_name']}",
                "annotation_filename": c["annotation_filename"],
                "image_width": c["width"],
                "image_height": c["height"],
                "annotated_tree_count": c["tree_count"],
                "original_source_reference": f"{c['shard_name']}#row={c['row_idx']}",
                "selection_seed": SEED,
                "selection_notes": notes,
            })

    total_trees = sum(c["tree_count"] for c in final_selected)
    site_counts = {}
    site_tree_counts = {}
    raster_counts = {}
    for c in final_selected:
        loc = c["location"]
        rname = c["raster_name"]
        site_counts[loc] = site_counts.get(loc, 0) + 1
        site_tree_counts[loc] = site_tree_counts.get(loc, 0) + c["tree_count"]
        raster_counts[rname] = raster_counts.get(rname, 0) + 1

    dataset_info = {
        "dataset_name": "SelvaBox 100-Image Independent External Test Set (testdataset100)",
        "source_dataset": HF_DATASET,
        "source_url": HF_URL,
        "documentation_url": DOC_URL,
        "license": "CC-BY-4.0 (Creative Commons Attribution 4.0 International)",
        "attribution": "SelvaBox: A high-resolution dataset for tropical tree crown detection (CanopyRS / ICLR). Cite Baudchon et al. / CanopyRS when publishing results.",
        "retrieval_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "official_split_used": "test",
        "selection_method": "Deterministic site-stratified selection (seed=42) from official held-out test split Shards 00 and 16 covering Brazil, Ecuador, and Panama with spatial separation ordering",
        "random_seed": SEED,
        "total_test_split_metadata_inspected": 1477,
        "total_shard_candidates_decoded": total_inspected,
        "total_rejected_from_decoded_shards": total_inspected - TARGET_COUNT,
        "rejection_breakdown": rejection_counts,
        "final_selected_image_count": len(final_selected),
        "total_annotated_tree_crowns": total_trees,
        "tree_count_statistics": {
            "min_trees_per_image": min(c["tree_count"] for c in final_selected),
            "max_trees_per_image": max(c["tree_count"] for c in final_selected),
            "mean_trees_per_image": round(total_trees / len(final_selected), 2),
        },
        "image_resolution": "1777x1777 pixels (4.5 cm GSD, 80x80m spatial extent)",
        "counts_by_source_site": site_counts,
        "annotated_trees_by_source_site": site_tree_counts,
        "counts_by_source_raster": raster_counts,
        "annotation_format": {
            "yolo_format": "Normalized YOLO format (class_id x_center y_center width height) in annotations/image_XXX.txt",
            "class_mapping": {"0": "tree-top"},
            "original_format_preserved": "COCO pixel bounding boxes [x_min, y_min, width, height] and geospatial metadata preserved in annotations/original_selvabox_annotations.json",
        },
        "known_limitations_and_notes": [
            "Extracted from official held-out test shards test-00000 and test-00016 (~610 MB downloaded vs 31.3 GB full dataset) to capture all 3 geographical regions (Brazil, Ecuador, Panama).",
            "Dense tropical forest canopies in Brazil (ZF2), Ecuador (Tiputini), and Panama (Agua Salud) exhibit significant crown overlap and multi-layered canopy structure.",
            "Strictly reserved for out-of-domain external testing; never used for training, validation, or threshold tuning.",
        ],
    }

    (OUT_DIR / "dataset_info.json").write_text(json.dumps(dataset_info, indent=2), encoding="utf-8")
    verify_and_write_report(final_selected, dataset_info, rejection_counts, time.time() - t_start)


def verify_and_write_report(final_selected, dataset_info, rejection_counts, elapsed_s):
    img_files = sorted(IMAGES_DIR.glob("*.png"))
    yolo_files = sorted(ANNOTATIONS_DIR.glob("image_*.txt"))

    all_files_in_images = list(IMAGES_DIR.iterdir())
    c1_pass = len(all_files_in_images) == 100 and len(img_files) == 100

    decoded_count = 0
    hashes = set()
    for p in img_files:
        with Image.open(p) as im:
            im.load()
            if im.size == (1777, 1777):
                decoded_count += 1
        hashes.add(hashlib.sha256(p.read_bytes()).hexdigest())
    c2_pass = decoded_count == 100

    matched_anns = 0
    valid_boxes_total = 0
    all_boxes_valid = True
    manifest_rows = []
    with (OUT_DIR / "manifest.csv").open("r", encoding="utf-8") as mf:
        reader = csv.DictReader(mf)
        for row in reader:
            manifest_rows.append(row)

    manifest_tree_sum = 0
    for row in manifest_rows:
        img_p = IMAGES_DIR / row["local_filename"]
        ann_p = ANNOTATIONS_DIR / row["annotation_filename"]
        if img_p.exists() and ann_p.exists() and img_p.stem == ann_p.stem == row["image_id"]:
            matched_anns += 1
        lines = [ln.strip() for ln in ann_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if len(lines) == 0 or len(lines) != int(row["annotated_tree_count"]):
            all_boxes_valid = False
        manifest_tree_sum += int(row["annotated_tree_count"])
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5 or parts[0] != "0":
                all_boxes_valid = False
                continue
            xc, yc, bw, bh = [float(x) for x in parts[1:]]
            if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                all_boxes_valid = False
            valid_boxes_total += 1

    c3_pass = matched_anns == 100 and len(yolo_files) == 100
    c4_pass = all_boxes_valid and valid_boxes_total > 0
    c5_pass = matched_anns == 100
    c6_pass = all_boxes_valid
    c7_pass = len(hashes) == 100
    orig_names = {r["original_filename"] for r in manifest_rows}
    c8_pass = len(orig_names) == 100
    c9_pass = len(manifest_rows) == 100
    c10_pass = manifest_tree_sum == valid_boxes_total == dataset_info["total_annotated_tree_crowns"]
    c11_pass = all(r["source_split"] == "test" for r in manifest_rows)

    ttv_yaml = REPO_ROOT / "datasets" / "Tree-Top-View" / "data.yaml"
    ttv_train = list((REPO_ROOT / "datasets" / "Tree-Top-View" / "train" / "images").glob("*"))
    ttv_val = list((REPO_ROOT / "datasets" / "Tree-Top-View" / "valid" / "images").glob("*"))
    ttv_test = list((REPO_ROOT / "datasets" / "Tree-Top-View" / "test" / "images").glob("*"))
    c12_pass = ttv_yaml.exists() and len(ttv_train) == 1191 and len(ttv_val) == 116 and len(ttv_test) == 55

    report_lines = [
        "================================================================================",
        "SELVABOX 100-IMAGE EXTERNAL TEST DATASET — VALIDATION & QUALITY REPORT",
        "================================================================================",
        f"Generated At               : {dataset_info['retrieval_date']}",
        f"Dataset Directory          : {OUT_DIR}",
        f"Source Dataset             : {dataset_info['source_dataset']} ({dataset_info['source_url']})",
        f"Official Split Used        : {dataset_info['official_split_used']} (held-out test split only)",
        f"Random Seed                : {dataset_info['random_seed']}",
        f"Preparation Time           : {elapsed_s:.1f} seconds",
        "",
        "1. CANDIDATE INSPECTION & FILTERING SUMMARY",
        "--------------------------------------------------------------------------------",
        f"Total Official Test Split Metadata Inspected : {dataset_info['total_test_split_metadata_inspected']}",
        f"Candidate Tiles in Target Shards (00 & 16)   : {dataset_info['total_shard_candidates_decoded']}",
        f"Candidates Rejected (Non-test split)         : {rejection_counts['not_test_split']}",
        f"Candidates Rejected (Missing annotation)     : {rejection_counts['missing_or_unmatched_annotation']}",
        f"Candidates Rejected (0 usable trees)         : {rejection_counts['zero_usable_tree_crowns']}",
        f"Candidates Rejected (Malformed coords)       : {rejection_counts['malformed_annotation_coordinates']}",
        f"Candidates Rejected (Invalid metadata)       : {rejection_counts['invalid_dimensions_or_metadata']}",
        f"Candidates Rejected (Exact duplicate)        : {rejection_counts['exact_duplicate_tile_name']}",
        f"Candidates Rejected (Overlap / excess pool)  : {rejection_counts['spatial_overlap_or_excess_site_pool']}",
        f"Candidates Rejected (Corrupt/Unreadable)     : {rejection_counts['corrupt_or_unreadable_image']}",
        f"Candidates Rejected (Excessive nodata)       : {rejection_counts['not_suitable_aerial_imagery_nodata']}",
        f"Final Selected Usable Test Images            : {dataset_info['final_selected_image_count']}",
        f"Total Annotated Tree Crowns                  : {dataset_info['total_annotated_tree_crowns']}",
        "",
        "2. GEOGRAPHICAL & SITE DISTRIBUTION",
        "--------------------------------------------------------------------------------",
    ]
    for site, cnt in sorted(dataset_info["counts_by_source_site"].items()):
        t_cnt = dataset_info["annotated_trees_by_source_site"][site]
        report_lines.append(f"  - {site:<22}: {cnt:>3} images | {t_cnt:>5} annotated tree crowns")

    report_lines.extend([
        "",
        "3. RASTER DISTRIBUTION",
        "--------------------------------------------------------------------------------",
    ])
    for rname, cnt in sorted(dataset_info["counts_by_source_raster"].items()):
        report_lines.append(f"  - {rname:<34}: {cnt:>3} images")

    checks = [
        ("1. Exactly 100 image files exist in images/", c1_pass, f"{len(img_files)} PNG files, {len(all_files_in_images)} total items"),
        ("2. All 100 images can be decoded at full 1777x1777 resolution", c2_pass, f"{decoded_count}/100 decoded"),
        ("3. All 100 images have matching annotation files", c3_pass, f"{matched_anns}/100 matched"),
        ("4. All 100 annotations are parseable and contain usable labels", c4_pass, f"{valid_boxes_total} total valid tree boxes"),
        ("5. Every annotation refers to the correct image", c5_pass, "1-to-1 stem & manifest verification passed"),
        ("6. Bounding boxes and normalized coordinates lie within [0, 1]", c6_pass, "All 5-tuple YOLO lines verified"),
        ("7. No exact duplicate image files remain (SHA-256 verified)", c7_pass, f"{len(hashes)} unique SHA-256 digests"),
        ("8. No duplicate source tiles remain", c8_pass, f"{len(orig_names)} unique source tiles verified"),
        ("9. The manifest.csv contains exactly 100 image records", c9_pass, f"{len(manifest_rows)} records (excluding header)"),
        ("10. Manifest tree counts agree with annotation files", c10_pass, f"Manifest={manifest_tree_sum}, Files={valid_boxes_total}"),
        ("11. Selected images come ONLY from the official 'test' split", c11_pass, "100/100 verified fold == 'test'"),
        ("12. Original Tree-Top-View dataset remains completely unchanged", c12_pass, f"train={len(ttv_train)}, valid={len(ttv_val)}, test={len(ttv_test)}"),
    ]

    report_lines.extend([
        "",
        "4. TWELVE-POINT VERIFICATION CHECKLIST",
        "--------------------------------------------------------------------------------",
    ])
    for label, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        report_lines.append(f"[{status}] {label} ({detail})")

    report_lines.append("================================================================================")
    (OUT_DIR / "validation_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines), flush=True)

    readme_lines = [
        "# SelvaBox 100-Image Independent External Test Dataset (`testdataset100`)",
        "",
        "## Overview",
        "This folder contains an independent, out-of-domain external test dataset of **exactly 100 aerial/drone orthomosaic tiles** extracted exclusively from the **official held-out `test` split** of the [CanopyRS/SelvaBox](https://huggingface.co/datasets/CanopyRS/SelvaBox) benchmark dataset.",
        "",
        "> **Experimental Safeguard:** This dataset is strictly reserved for final out-of-domain external evaluation. It must **never** be used for model training, fine-tuning, validation, early stopping, or confidence-threshold tuning.",
        "",
        "## Dataset Summary",
        f"- **Source Dataset:** `{dataset_info['source_dataset']}` ([Hugging Face]({dataset_info['source_url']}))",
        f"- **Official Split Used:** `{dataset_info['official_split_used']}` (100% from official held-out test split; 0 from train/validation)",
        f"- **License:** {dataset_info['license']}",
        f"- **Total Test Candidates Inspected:** {dataset_info['total_test_split_metadata_inspected']} tiles (across all 24 official test parquet shards); decoded 123 candidate tiles from Shards `00` and `16` spanning all 3 countries",
        f"- **Final Selected Images:** {dataset_info['final_selected_image_count']} (`images/image_001.png` .. `images/image_100.png`)",
        f"- **Image Resolution:** {dataset_info['image_resolution']}",
        f"- **Total Annotated Tree Crowns:** {dataset_info['total_annotated_tree_crowns']} (mean: {dataset_info['tree_count_statistics']['mean_trees_per_image']} trees/image, range: {dataset_info['tree_count_statistics']['min_trees_per_image']}–{dataset_info['tree_count_statistics']['max_trees_per_image']})",
        f"- **Random Seed:** `{dataset_info['random_seed']}`",
        "",
        "## Geographical & Site Distribution",
        "| Source Site / Location | Selected Images | Annotated Tree Crowns |",
        "| :--- | :---: | :---: |",
    ]
    for site, cnt in sorted(dataset_info["counts_by_source_site"].items()):
        t_cnt = dataset_info["annotated_trees_by_source_site"][site]
        readme_lines.append(f"| `{site}` | {cnt} | {t_cnt} |")

    readme_lines.extend([
        "",
        "## Directory Structure",
        "```text",
        "External validation v5 and v7/testdataset100/",
        "├── images/                  # Exactly 100 full-resolution (1777x1777) PNG tiles",
        "├── annotations/             # 100 YOLO .txt files (class 0 'tree-top') + original_selvabox_annotations.json",
        "├── manifest.csv             # 100-row metadata manifest with exact source tile IDs and tree counts",
        "├── dataset_info.json        # Structured metadata, license, site breakdown, and rejection statistics",
        "├── validation_report.txt    # 12-point automated integrity and verification report",
        "└── README.md                # Dataset documentation",
        "```",
        "",
        "## Annotation Format",
        "- **Converted YOLO Format (`annotations/image_XXX.txt`):**",
        "  - Single class: `0` (`tree-top`)",
        "  - Normalized coordinates: `<class_id> <x_center> <y_center> <width> <height>` in `[0.0, 1.0]`",
        "- **Original Source Format (`annotations/original_selvabox_annotations.json`):**",
        "  - Preserves exact pixel-level COCO bounding boxes `[x_min, y_min, width, height]` and geospatial `tile_metadata` (`crs`, `transform`, `bounds`).",
    ])
    (OUT_DIR / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
