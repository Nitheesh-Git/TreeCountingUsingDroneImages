import os
import glob
import random
import json
import yaml

def main():
    seed = 42
    target_count = 595
    train_img_dir = "datasets/Tree-Top-View/train/images"
    train_lbl_dir = "datasets/Tree-Top-View/train/labels"

    # 1. Deterministic sort of all train images
    all_imgs = sorted(glob.glob(os.path.join(train_img_dir, "*.*")))
    total_train = len(all_imgs)
    print(f"Total training images found: {total_train}")

    # 2. Reproducible random sampling with fixed seed
    rng = random.Random(seed)
    # Sort after sample to keep manifest clean and sorted
    sampled_imgs = sorted(rng.sample(all_imgs, target_count))
    print(f"Sampled images count: {len(sampled_imgs)} using seed={seed}")

    # 3. Analyze subset and full set statistics
    def analyze_set(img_list):
        total_instances = 0
        empty_images = 0
        box_widths = []
        box_heights = []
        box_areas = []

        for img_path in img_list:
            fname = os.path.basename(img_path)
            base, _ = os.path.splitext(fname)
            lbl_path = os.path.join(train_lbl_dir, base + ".txt")
            if not os.path.exists(lbl_path):
                raise FileNotFoundError(f"Missing label file for: {img_path}")

            with open(lbl_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

            if len(lines) == 0:
                empty_images += 1
            else:
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        total_instances += 1
                        w = float(parts[3])
                        h = float(parts[4])
                        box_widths.append(w)
                        box_heights.append(h)
                        box_areas.append(w * h)

        avg_w = sum(box_widths) / len(box_widths) if box_widths else 0.0
        avg_h = sum(box_heights) / len(box_heights) if box_heights else 0.0
        avg_area = sum(box_areas) / len(box_areas) if box_areas else 0.0

        return {
            "image_count": len(img_list),
            "total_instances": total_instances,
            "background_images": empty_images,
            "avg_instances_per_image": total_instances / len(img_list) if img_list else 0.0,
            "avg_box_width": avg_w,
            "avg_box_height": avg_h,
            "avg_box_area": avg_area,
        }

    full_stats = analyze_set(all_imgs)
    subset_stats = analyze_set(sampled_imgs)

    print("Full Training Set Stats:", full_stats)
    print("595-Image Subset Stats :", subset_stats)

    # 4. Save manifest files
    # Manifest 1: Relative to datasets/Tree-Top-View
    manifest_rel_dataset = "datasets/Tree-Top-View/train_subset_595.txt"
    with open(manifest_rel_dataset, "w") as f:
        for img in sampled_imgs:
            # normalize to ./train/images/xxxx.jpg so Ultralytics/YOLOv7 resolve relative to manifest dir
            rel = "./" + os.path.relpath(img, "datasets/Tree-Top-View").replace("\\", "/")
            f.write(rel + "\n")

    # Manifest 2: In External validation v5 and v7 for documentation and inspection
    manifest_doc = "External validation v5 and v7/subset_595_manifest.txt"
    with open(manifest_doc, "w") as f:
        for img in sampled_imgs:
            f.write(os.path.basename(img) + "\n")

    # 5. Create YAML configuration for subset
    # Uses absolute path or path relative to External validation v5 and v7
    subset_yaml_path = "External validation v5 and v7/data_subset_595.yaml"
    repo_root = os.path.abspath(".")
    dataset_root = os.path.abspath("datasets/Tree-Top-View")
    
    yaml_content = {
        "path": dataset_root.replace("\\", "/"),
        "train": "train_subset_595.txt",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["tree-top"],
        "metadata": {
            "subset_name": "train_subset_595",
            "seed": seed,
            "fraction": 0.4996,
            "description": "Reproducible 595-image subset for comparison models"
        }
    }

    with open(subset_yaml_path, "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)

    # 6. Save stats summary
    stats_summary = {
        "seed": seed,
        "subset_count": target_count,
        "full_train_count": total_train,
        "full_stats": full_stats,
        "subset_stats": subset_stats
    }
    with open("External validation v5 and v7/subset_statistics.json", "w") as f:
        json.dump(stats_summary, f, indent=2)

    print(f"\n[SUCCESS] Manifest created at: {manifest_rel_dataset}")
    print(f"[SUCCESS] Dataset YAML created at: {subset_yaml_path}")
    print(f"[SUCCESS] Statistics saved at: External validation v5 and v7/subset_statistics.json")

if __name__ == "__main__":
    main()
