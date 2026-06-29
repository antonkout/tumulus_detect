"""
tumulus_detect.py
=================
Run trained YOLOv8 model on local Pleiades GeoTIFF images.
Outputs detections as GeoPackage + visualizations.

Usage:
    python tumulus_detect.py

Configure paths and parameters in the CONFIGURATION section below.
"""

import os
import sys
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol, xy
import geopandas as gpd
from shapely.geometry import Point, box
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — edit these paths and parameters
# ============================================================

# Path to your trained model (downloaded from Google Drive)
MODEL_PATH = "./models/1_colab4_best.pt"

# Your local Pleiades TIF files
IMAGE_PATHS = [
    "./data/pleiades1.TIF",
    "./data/pleiades3.TIF",
    # add more if needed
]

# Optional: test labels GeoPackage to evaluate recall
# Set to None to skip evaluation
TEST_LABELS_PATH = "./data/tombs_testing.gpkg"

# Output directory for detections
OUTPUT_DIR = "./output/detections"

# ============================================================
# DETECTION PARAMETERS
# ============================================================

PIXEL_SIZE       = 0.15    # Pleiades Neo HD pixel size in meters
TUMULUS_DIAMETER = 10      # Expected tumulus diameter in meters (for visualization)
PATCH_SIZE       = 640     # Must match training patch size
OVERLAP          = 160     # Overlap between tiles (higher = fewer missed edge cases)

# Confidence threshold — lower = more detections, more false positives
# Start at 0.15 for archaeological survey (better recall)
# Raise to 0.30+ if too many false positives
CONF_THRESHOLD   = 0.15

# IoU threshold for NMS (non-maximum suppression)
IOU_THRESHOLD    = 0.4

# NMS suppression distance in pixels (detections closer than this are merged)
# Set to tumulus diameter in pixels
NMS_DIST_PX      = int(TUMULUS_DIAMETER / PIXEL_SIZE)

# Test-time augmentation — runs model on flips/rotations, improves recall ~5-10%
USE_TTA          = True

# ============================================================


def normalize_rgb(data, nodata_val=0):
    """Stretch 3-band raster data to uint8 RGB using percentile normalization."""
    rgb = np.stack([data[0], data[1], data[2]], axis=-1).astype(np.float32)
    for ch in range(3):
        vals = rgb[:, :, ch]
        valid = vals[vals > nodata_val]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, [2, 98])
            rgb[:, :, ch] = np.clip((vals - p2) / (p98 - p2 + 1e-10) * 255, 0, 255)
    return rgb.astype(np.uint8)


def find_image_for_point(image_paths, x, y):
    """Return which image path contains the given geographic coordinate."""
    for img_path in image_paths:
        with rasterio.open(img_path) as src:
            b = src.bounds
            if b.left <= x <= b.right and b.bottom <= y <= b.top:
                return img_path
    return None


def predict_with_tta(model, rgb, conf, iou, patch_size):
    """
    Run inference with test-time augmentation.
    Tries 4 rotations + horizontal flip, merges all detections before NMS.
    """
    from ultralytics import YOLO

    all_boxes = []  # list of [x1, y1, x2, y2, conf]

    augmentations = [
        (lambda img: img,                           lambda b, s: b),                          # original
        (lambda img: np.rot90(img, 1).copy(),       lambda b, s: _rot90_box(b, s, 1)),        # 90°
        (lambda img: np.rot90(img, 2).copy(),       lambda b, s: _rot90_box(b, s, 2)),        # 180°
        (lambda img: np.rot90(img, 3).copy(),       lambda b, s: _rot90_box(b, s, 3)),        # 270°
        (lambda img: np.fliplr(img).copy(),         lambda b, s: _fliplr_box(b, s)),          # flip LR
    ]

    for aug_fn, inv_fn in augmentations:
        aug_img = aug_fn(rgb)
        results = model.predict(aug_img, conf=conf, iou=iou, verbose=False, imgsz=patch_size)
        for result in results:
            if result.boxes is not None:
                for bx in result.boxes:
                    coords = bx.xyxy[0].cpu().numpy()
                    c = bx.conf[0].cpu().item()
                    inv_coords = inv_fn(coords, patch_size)
                    all_boxes.append((*inv_coords, c))

    return all_boxes


def _rot90_box(box, size, k):
    """Inverse-transform a bounding box after k*90° rotation."""
    x1, y1, x2, y2 = box
    s = size
    for _ in range(k):
        # inverse of rot90 by 1 is rot90 by 3
        x1, y1, x2, y2 = y1, s - x2, y2, s - x1
    return x1, y1, x2, y2


def _fliplr_box(box, size):
    x1, y1, x2, y2 = box
    return size - x2, y1, size - x1, y2


def pixel_nms(detections, nms_dist_px):
    """
    Simple distance-based NMS in pixel space.
    Keeps highest-confidence detection when two are within nms_dist_px.
    """
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    keep = []
    suppressed = set()
    for i, di in enumerate(dets):
        if i in suppressed:
            continue
        keep.append(di)
        for j, dj in enumerate(dets[i+1:], start=i+1):
            if j in suppressed:
                continue
            dist = np.sqrt((di['row'] - dj['row'])**2 + (di['col'] - dj['col'])**2)
            if dist < nms_dist_px:
                suppressed.add(j)
    return keep


def detect_image(image_path, model, conf_threshold, iou_threshold,
                 patch_size, overlap, pixel_size, nms_dist_px, use_tta):
    """
    Sliding-window detection on a full Pleiades GeoTIFF.
    Returns list of detection dicts with geographic coordinates.
    """
    with rasterio.open(image_path) as src:
        img_h, img_w = src.height, src.width
        crs = src.crs
        transform = src.transform

    stride = patch_size - overlap
    n_tiles_y = max(1, int(np.ceil((img_h - overlap) / stride)))
    n_tiles_x = max(1, int(np.ceil((img_w - overlap) / stride)))
    total_tiles = n_tiles_y * n_tiles_x

    print(f"\n  Image: {Path(image_path).name}  ({img_w}×{img_h} px)")
    print(f"  Tiles: {n_tiles_x}×{n_tiles_y} = {total_tiles}  |  stride={stride}px  |  TTA={'on' if use_tta else 'off'}")

    raw_detections = []
    tile_count = 0

    with rasterio.open(image_path) as src:
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_count += 1

                row_off = min(ty * stride, img_h - patch_size)
                col_off = min(tx * stride, img_w - patch_size)
                row_off = max(0, row_off)
                col_off = max(0, col_off)

                win_h = min(patch_size, img_h - row_off)
                win_w = min(patch_size, img_w - col_off)
                window = Window(col_off, row_off, win_w, win_h)
                data = src.read(window=window)

                rgb = normalize_rgb(data)

                # Pad to patch_size if edge tile is smaller
                if win_h < patch_size or win_w < patch_size:
                    padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                    padded[:win_h, :win_w] = rgb
                    rgb = padded

                # Run inference
                if use_tta:
                    boxes = predict_with_tta(model, rgb, conf_threshold, iou_threshold, patch_size)
                else:
                    results = model.predict(rgb, conf=conf_threshold, iou=iou_threshold,
                                           verbose=False, imgsz=patch_size)
                    boxes = []
                    for result in results:
                        if result.boxes is not None:
                            for bx in result.boxes:
                                coords = bx.xyxy[0].cpu().numpy()
                                c = bx.conf[0].cpu().item()
                                boxes.append((*coords, c))

                for (x1, y1, x2, y2, conf) in boxes:
                    # Convert patch-local pixel coords to image pixel coords
                    center_col = col_off + (x1 + x2) / 2
                    center_row = row_off + (y1 + y2) / 2

                    # Skip detections in padded area
                    if center_row >= img_h or center_col >= img_w:
                        continue

                    geo_x, geo_y = xy(transform, int(center_row), int(center_col))

                    raw_detections.append({
                        'geometry':    Point(geo_x, geo_y),
                        'confidence':  round(float(conf), 3),
                        'width_m':     round((x2 - x1) * pixel_size, 2),
                        'height_m':    round((y2 - y1) * pixel_size, 2),
                        'diameter_m':  round(((x2 - x1) + (y2 - y1)) / 2 * pixel_size, 2),
                        'row':         int(center_row),
                        'col':         int(center_col),
                        'source':      Path(image_path).name,
                    })

                if tile_count % 500 == 0:
                    pct = tile_count / total_tiles * 100
                    print(f"    [{pct:5.1f}%] {tile_count}/{total_tiles} tiles — {len(raw_detections)} raw detections")

    print(f"  Raw detections: {len(raw_detections)}")

    # NMS
    kept = pixel_nms(raw_detections, nms_dist_px)
    print(f"  After NMS ({nms_dist_px}px): {len(kept)} detections")

    return kept, crs


def evaluate_against_labels(detections_gdf, test_labels_path, image_paths,
                             pixel_size, tumulus_diameter, patch_size, model,
                             output_dir, conf_threshold, iou_threshold):
    """
    Visualize model predictions at test label locations.
    """
    test_gdf = gpd.read_file(test_labels_path)
    print(f"\nEvaluating against {len(test_gdf)} test labels...")

    n_test = len(test_gdf)
    ncols = min(4, n_test)
    nrows = max(1, (n_test + ncols - 1) // ncols)
    bbox_px = int(tumulus_diameter / pixel_size)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n_test > 1 else [axes]

    detected = 0
    total = 0

    for idx in range(n_test):
        row = test_gdf.iloc[idx]
        x, y = row.geometry.x, row.geometry.y
        img_path = find_image_for_point(image_paths, x, y)
        if img_path is None:
            axes[idx].set_title(f"#{idx}: not in any image", color='gray')
            axes[idx].axis('off')
            continue

        total += 1

        with rasterio.open(img_path) as src:
            pr, pc = rowcol(src.transform, x, y)
            pr, pc = int(pr), int(pc)
            half = patch_size // 2
            r_off = int(np.clip(pr - half, 0, src.height - patch_size))
            c_off = int(np.clip(pc - half, 0, src.width - patch_size))
            window = Window(c_off, r_off, patch_size, patch_size)
            data = src.read(window=window)

        rgb = normalize_rgb(data)

        results = model.predict(rgb, conf=conf_threshold, iou=iou_threshold,
                               verbose=False, imgsz=patch_size)

        axes[idx].imshow(rgb)

        # Ground truth circle
        gt_cy = pr - r_off
        gt_cx = pc - c_off
        circle = plt.Circle((gt_cx, gt_cy), bbox_px // 2,
                            fill=False, color='lime', linewidth=2,
                            linestyle='--', label='GT')
        axes[idx].add_patch(circle)

        is_detected = False
        n_det = 0
        for result in results:
            if result.boxes is not None:
                for bx in result.boxes:
                    x1, y1, x2, y2 = bx.xyxy[0].cpu().numpy()
                    conf = bx.conf[0].cpu().item()
                    rect = mpatches.Rectangle(
                        (x1, y1), x2 - x1, y2 - y1,
                        linewidth=2, edgecolor='red', facecolor='none'
                    )
                    axes[idx].add_patch(rect)
                    axes[idx].text(x1, max(0, y1 - 5), f'{conf:.2f}',
                                  color='red', fontsize=8, fontweight='bold',
                                  backgroundcolor='white')
                    n_det += 1
                    det_cx = (x1 + x2) / 2
                    det_cy = (y1 + y2) / 2
                    dist = np.sqrt((det_cx - gt_cx)**2 + (det_cy - gt_cy)**2)
                    if dist < bbox_px:
                        is_detected = True

        if is_detected:
            detected += 1

        status = "✓ DETECTED" if is_detected else "✗ MISSED"
        color = 'green' if is_detected else 'red'
        axes[idx].set_title(f"#{idx}: {status}  ({n_det} det)", color=color, fontsize=10)
        axes[idx].axis('off')

    for idx in range(n_test, len(axes)):
        axes[idx].axis('off')

    recall = detected / total if total > 0 else 0
    plt.suptitle(
        f"Test Evaluation — {detected}/{total} detected  (recall = {recall:.1%})\n"
        f"conf={conf_threshold}  |  Green dashed = ground truth  |  Red = YOLO detection",
        fontsize=12, y=1.01
    )
    plt.tight_layout()
    eval_path = os.path.join(output_dir, 'evaluation_test_labels.png')
    plt.savefig(eval_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\n{'='*50}")
    print(f"  RECALL: {detected}/{total} = {recall:.1%}")
    print(f"  Evaluation plot saved: {eval_path}")
    print(f"{'='*50}")


def visualize_top_detections(detections_gdf, image_paths, pixel_size,
                             tumulus_diameter, patch_size, output_dir, n=16):
    """Show a grid of the highest-confidence detections."""
    top = detections_gdf.nlargest(min(n, len(detections_gdf)), 'confidence')
    ncols = 4
    nrows = max(1, (len(top) + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = np.array(axes).flatten()

    for idx, (_, det) in enumerate(top.iterrows()):
        if idx >= len(axes):
            break
        x, y = det.geometry.x, det.geometry.y
        img_path = find_image_for_point(image_paths, x, y)
        if img_path is None:
            axes[idx].axis('off')
            continue

        with rasterio.open(img_path) as src:
            pr, pc = rowcol(src.transform, x, y)
            pr, pc = int(pr), int(pc)
            view_size = 300
            half = view_size // 2
            r_off = int(np.clip(pr - half, 0, src.height - view_size))
            c_off = int(np.clip(pc - half, 0, src.width - view_size))
            window = Window(c_off, r_off, view_size, view_size)
            data = src.read(window=window)

        rgb = normalize_rgb(data)
        cx = pc - c_off
        cy = pr - r_off

        axes[idx].imshow(rgb)
        r_px = int(det['diameter_m'] / pixel_size / 2)
        circle = plt.Circle((cx, cy), r_px, fill=False, color='red', linewidth=2)
        axes[idx].add_patch(circle)
        axes[idx].set_title(
            f"conf={det['confidence']:.2f}  d≈{det['diameter_m']:.0f}m",
            fontsize=9
        )
        axes[idx].axis('off')

    for idx in range(len(top), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Top {len(top)} Detections by Confidence", fontsize=13)
    plt.tight_layout()
    viz_path = os.path.join(output_dir, 'top_detections.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {viz_path}")


def plot_confidence_distribution(detections_gdf, tumulus_diameter, output_dir):
    """Histogram of confidence scores and diameter estimates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(detections_gdf['confidence'], bins=20, edgecolor='black', color='steelblue')
    ax1.axvline(0.25, color='orange', linestyle='--', label='conf=0.25')
    ax1.axvline(0.50, color='red',    linestyle='--', label='conf=0.50')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Detection Confidence Distribution')
    ax1.legend()

    ax2.hist(detections_gdf['diameter_m'], bins=20, edgecolor='black', color='darkorange')
    ax2.axvline(tumulus_diameter, color='red', linestyle='--',
                label=f'Expected ({tumulus_diameter}m)')
    ax2.set_xlabel('Estimated Diameter (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Detected Object Size Distribution')
    ax2.legend()

    plt.tight_layout()
    dist_path = os.path.join(output_dir, 'detection_distribution.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {dist_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    from ultralytics import YOLO

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Validate paths
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Download best.pt from Google Drive and update MODEL_PATH above.")
        sys.exit(1)

    missing = [p for p in IMAGE_PATHS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Missing images: {missing}")
        sys.exit(1)

    print("=" * 60)
    print("  TUMULUS DETECTION")
    print("=" * 60)
    print(f"  Model:      {MODEL_PATH}")
    print(f"  Images:     {len(IMAGE_PATHS)}")
    print(f"  Conf:       {CONF_THRESHOLD}")
    print(f"  Overlap:    {OVERLAP}px")
    print(f"  TTA:        {'enabled' if USE_TTA else 'disabled'}")
    print("=" * 60)

    # Load model
    model = YOLO(MODEL_PATH)

    # Run detection on all images
    all_detections = []
    all_crs = None

    for img_path in IMAGE_PATHS:
        dets, crs = detect_image(
            img_path, model,
            conf_threshold=CONF_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            pixel_size=PIXEL_SIZE,
            nms_dist_px=NMS_DIST_PX,
            use_tta=USE_TTA,
        )
        all_detections.extend(dets)
        if all_crs is None:
            all_crs = crs

    if not all_detections:
        print("\nNo detections found. Try lowering CONF_THRESHOLD.")
        return

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(all_detections, crs=all_crs)
    gdf = gdf.drop(columns=['row', 'col'])  # pixel coords not needed in output
    gdf['id'] = range(1, len(gdf) + 1)

    # Save outputs
    gpkg_path = os.path.join(OUTPUT_DIR, 'tumulus_detections.gpkg')
    gdf.to_file(gpkg_path, driver='GPKG', layer='detections')

    # Also save confidence-filtered subsets
    for threshold in [0.25, 0.40, 0.60]:
        subset = gdf[gdf['confidence'] >= threshold]
        if len(subset) > 0:
            subset.to_file(
                os.path.join(OUTPUT_DIR, f'detections_conf{int(threshold*100)}.gpkg'),
                driver='GPKG', layer='detections'
            )

    print(f"\n{'='*60}")
    print(f"  TOTAL DETECTIONS: {len(gdf)}")
    print(f"  Saved to: {gpkg_path}")
    print(f"  Also saved confidence-filtered subsets (25%, 40%, 60%)")
    print(f"{'='*60}")
    print(gdf[['confidence', 'diameter_m', 'source']].describe())

    # Evaluate against test labels if available
    if TEST_LABELS_PATH and os.path.exists(TEST_LABELS_PATH):
        evaluate_against_labels(
            gdf, TEST_LABELS_PATH, IMAGE_PATHS,
            PIXEL_SIZE, TUMULUS_DIAMETER, PATCH_SIZE,
            model, OUTPUT_DIR, CONF_THRESHOLD, IOU_THRESHOLD
        )
    else:
        print("\nSkipping test evaluation (TEST_LABELS_PATH not set or not found).")

    # Visualizations
    visualize_top_detections(gdf, IMAGE_PATHS, PIXEL_SIZE,
                             TUMULUS_DIAMETER, PATCH_SIZE, OUTPUT_DIR)
    plot_confidence_distribution(gdf, TUMULUS_DIAMETER, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Load tumulus_detections.gpkg in QGIS for full spatial review.")


if __name__ == "__main__":
    main()
