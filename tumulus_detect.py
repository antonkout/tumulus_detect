"""
tumulus_detect.py  [v3]
=======================
Run trained YOLOv8 model on local Pleiades GeoTIFF images.
Outputs detections as GeoPackage + visualizations.

Changes from v2:
  - FIXED evaluate_label_set:
      * Proper geographic-distance matching (not patch-pixel coords)
      * Computes Precision, Recall, F1 at multiple thresholds
      * Returns metrics dict for programmatic use
      * Proper multi-combo NMS via pixel_nms (not crude bucketing)
  - Hard negative export: saves FP detections as GeoPackage for
    direct use as hard negatives in training
  - Confidence sweep: automatically finds optimal threshold via F1
  - Summary table printed at end

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
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — edit these paths and parameters
# ============================================================

MODEL_PATH = "./models/3_colab_aggresive8_best.pt"

IMAGE_PATHS = [
    # "./data/pleiades1.TIF",
    # "./data/pleiades2.TIF",
    "./data/pleiades3.TIF",
    # "./data/pleiades4.TIF",
]

# Held-out test set (never used for training)
TEST_LABELS_PATH  = "./data/tombs_testing.gpkg"

# Validation set (used to monitor training, now also used for spatial QA)
VAL_LABELS_PATH   = None   # set to None to skip

# Training set — evaluate to check for overfitting / data leakage
TRAIN_LABELS_PATH = "./data/tombs_training.gpkg"      # set to None to skip

OUTPUT_DIR = "./output/detections"

# ============================================================
# DETECTION PARAMETERS
# ============================================================

PIXEL_SIZE       = 0.15    # Pleiades Neo HD pixel size in metres
TUMULUS_DIAMETER = 10      # Expected tumulus diameter in metres (for visualisation)
PATCH_SIZE       = 640     # Must match training imgsz
OVERLAP          = 160     # Tile overlap in pixels

CONF_THRESHOLD   = 0.15    # Lower = more recall, more false positives
IOU_THRESHOLD    = 0.4     # NMS IoU threshold
NMS_DIST_PX      = int(TUMULUS_DIAMETER / PIXEL_SIZE)   # pixel-space NMS radius

# Skip tiles where more than this fraction of pixels is nodata (black)
NODATA_SKIP_THRESHOLD = 0.15

# Test-time augmentation — 4 rotations + flip, merges before NMS
USE_TTA = True

# Band combinations to run inference on.
INFERENCE_COMBOS = ['rgb', 'cir', 'ndvi']

# Pleiades Neo band order (0-based), per DIMAP metadata (RGBN): Red=0, Green=1, Blue=2, NIR=3
BAND_ORDER = {'red': 0, 'green': 1, 'blue': 2, 'nir': 3}

# ── Matching radius for evaluation ──────────────────────────────────────────────
# A detection is a TP if its centre is within this distance (metres) of a GT point.
# Set to ~1x tumulus diameter. Larger = more lenient matching.
MATCH_RADIUS_M = 15.0

# ── Confidence thresholds for filtered exports ──────────────────────────────────
EXPORT_CONF_THRESHOLDS = [0.25, 0.40, 0.60]

# ============================================================


# ── Band normalisation ──────────────────────────────────────────────────────────

def normalize_to_uint8(data, band_indices, nodata_val=0):
    """Extract bands by index and stretch to uint8 via percentile normalisation."""
    rgb = np.stack([data[i] for i in band_indices], axis=-1).astype(np.float32)
    for ch in range(3):
        vals = rgb[:, :, ch]
        valid = vals[vals > nodata_val]
        if len(valid) > 100:
            p2, p98 = np.percentile(valid, [2, 98])
            rgb[:, :, ch] = np.clip((vals - p2) / (p98 - p2 + 1e-10) * 255, 0, 255)
    return rgb.astype(np.uint8)


def make_combo_images(data, n_bands, combos, band_order):
    """
    Build list of (combo_name, uint8_rgb) from raw raster data.
    Only generates combos that are both requested and possible given n_bands.
    """
    b = band_order
    results = []

    if 'rgb' in combos and n_bands >= 3:
        rgb = normalize_to_uint8(data, (b['red'], b['green'], b['blue']))
        results.append(('rgb', rgb))

    if n_bands >= 4:
        if 'cir' in combos:
            cir = normalize_to_uint8(data, (b['nir'], b['red'], b['green']))
            results.append(('cir', cir))

        if 'ndvi' in combos:
            nir = data[b['nir']].astype(np.float32)
            red = data[b['red']].astype(np.float32)
            ndvi = np.where((nir + red) > 0, (nir - red) / (nir + red + 1e-10), 0)
            ndvi_u8 = ((ndvi + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            rg_base = normalize_to_uint8(data, (b['red'], b['green'], b['blue']))
            ndvi_img = rg_base.copy()
            ndvi_img[:, :, 2] = ndvi_u8
            results.append(('ndvi', ndvi_img))

    if not results:
        rgb = normalize_to_uint8(data, (0, 1, 2))
        results.append(('rgb', rgb))

    return results


def nodata_fraction(data, nodata_val=0):
    """Fraction of pixels that are nodata (all bands == 0)."""
    mask = np.all(data <= nodata_val, axis=0)
    return mask.sum() / mask.size


# ── Utility ─────────────────────────────────────────────────────────────────────

def find_image_for_point(image_paths, x, y):
    for img_path in image_paths:
        with rasterio.open(img_path) as src:
            b = src.bounds
            if b.left <= x <= b.right and b.bottom <= y <= b.top:
                return img_path
    return None


# ── Test-time augmentation ───────────────────────────────────────────────────────

def _rot90_box(box, size, k):
    x1, y1, x2, y2 = box
    s = size
    for _ in range(k):
        x1, y1, x2, y2 = y1, s - x2, y2, s - x1
    return x1, y1, x2, y2


def _fliplr_box(box, size):
    x1, y1, x2, y2 = box
    return size - x2, y1, size - x1, y2


def predict_with_tta(model, rgb, conf, iou, patch_size):
    """Run 5 TTA variants, return merged raw box list [(x1,y1,x2,y2,conf)]."""
    augmentations = [
        (lambda img: img,                     lambda b, s: b),
        (lambda img: np.rot90(img, 1).copy(), lambda b, s: _rot90_box(b, s, 1)),
        (lambda img: np.rot90(img, 2).copy(), lambda b, s: _rot90_box(b, s, 2)),
        (lambda img: np.rot90(img, 3).copy(), lambda b, s: _rot90_box(b, s, 3)),
        (lambda img: np.fliplr(img).copy(),   lambda b, s: _fliplr_box(b, s)),
    ]
    all_boxes = []
    for aug_fn, inv_fn in augmentations:
        aug_img = aug_fn(rgb)
        res = model.predict(aug_img, conf=conf, iou=iou, verbose=False, imgsz=patch_size)
        for r in res:
            if r.boxes is not None:
                for bx in r.boxes:
                    coords = bx.xyxy[0].cpu().numpy()
                    c = bx.conf[0].cpu().item()
                    all_boxes.append((*inv_fn(coords, patch_size), c))
    return all_boxes


# ── NMS ─────────────────────────────────────────────────────────────────────────

def pixel_nms(detections, nms_dist_px):
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    keep, suppressed = [], set()
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


# ── Core detection ───────────────────────────────────────────────────────────────

def detect_image(image_path, model, conf_threshold, iou_threshold,
                 patch_size, overlap, pixel_size, nms_dist_px,
                 use_tta, inference_combos, band_order, nodata_skip_threshold):
    """
    Sliding-window detection on a full Pleiades GeoTIFF.
    Supports multi-band-combo inference and nodata tile skipping.
    Returns list of detection dicts.
    """
    with rasterio.open(image_path) as src:
        img_h, img_w = src.height, src.width
        n_bands = src.count
        crs = src.crs
        transform = src.transform

    stride = patch_size - overlap
    n_tiles_y = max(1, int(np.ceil((img_h - overlap) / stride)))
    n_tiles_x = max(1, int(np.ceil((img_w - overlap) / stride)))
    total_tiles = n_tiles_y * n_tiles_x

    active_combos = [c for c in inference_combos
                     if c == 'rgb' or n_bands >= 4]

    print(f"\n  Image:   {Path(image_path).name}  ({img_w}×{img_h} px, {n_bands} bands)")
    print(f"  Tiles:   {n_tiles_x}×{n_tiles_y} = {total_tiles}  |  stride={stride}px")
    print(f"  Combos:  {active_combos}  |  TTA={'on' if use_tta else 'off'}")

    raw_detections = []
    skipped_nodata = 0
    tile_count = 0

    with rasterio.open(image_path) as src:
        for ty in range(n_tiles_y):
            for tx in range(n_tiles_x):
                tile_count += 1

                row_off = int(np.clip(ty * stride, 0, img_h - patch_size))
                col_off = int(np.clip(tx * stride, 0, img_w - patch_size))

                win_h = min(patch_size, img_h - row_off)
                win_w = min(patch_size, img_w - col_off)
                window = Window(col_off, row_off, win_w, win_h)
                data = src.read(window=window)

                if nodata_fraction(data) > nodata_skip_threshold:
                    skipped_nodata += 1
                    continue

                combos = make_combo_images(data, n_bands, active_combos, band_order)

                padded_combos = []
                for combo_name, img in combos:
                    if win_h < patch_size or win_w < patch_size:
                        padded = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
                        padded[:win_h, :win_w] = img
                        img = padded
                    padded_combos.append((combo_name, img))

                tile_boxes = []
                for combo_name, img in padded_combos:
                    if use_tta:
                        boxes = predict_with_tta(model, img, conf_threshold,
                                                 iou_threshold, patch_size)
                    else:
                        res = model.predict(img, conf=conf_threshold,
                                           iou=iou_threshold, verbose=False,
                                           imgsz=patch_size)
                        boxes = []
                        for r in res:
                            if r.boxes is not None:
                                for bx in r.boxes:
                                    coords = bx.xyxy[0].cpu().numpy()
                                    c = bx.conf[0].cpu().item()
                                    boxes.append((*coords, c))
                    tile_boxes.extend(boxes)

                for (x1, y1, x2, y2, conf) in tile_boxes:
                    center_col = col_off + (x1 + x2) / 2
                    center_row = row_off + (y1 + y2) / 2
                    if center_row >= img_h or center_col >= img_w:
                        continue
                    geo_x, geo_y = xy(transform, int(center_row), int(center_col))
                    raw_detections.append({
                        'geometry':   Point(geo_x, geo_y),
                        'confidence': round(float(conf), 3),
                        'width_m':    round((x2 - x1) * pixel_size, 2),
                        'height_m':   round((y2 - y1) * pixel_size, 2),
                        'diameter_m': round(((x2-x1)+(y2-y1)) / 2 * pixel_size, 2),
                        'row':        int(center_row),
                        'col':        int(center_col),
                        'source':     Path(image_path).name,
                    })

                if tile_count % 500 == 0:
                    pct = tile_count / total_tiles * 100
                    print(f"    [{pct:5.1f}%] {tile_count}/{total_tiles} tiles  "
                          f"| skipped (nodata): {skipped_nodata}  "
                          f"| raw dets: {len(raw_detections)}")

    print(f"  Tiles skipped (nodata): {skipped_nodata}/{total_tiles}")
    print(f"  Raw detections: {len(raw_detections)}")
    kept = pixel_nms(raw_detections, nms_dist_px)
    print(f"  After NMS ({nms_dist_px}px): {len(kept)} detections")
    return kept, crs


# ── Evaluation (FIXED in v3) ─────────────────────────────────────────────────────

def match_detections_to_gt(det_gdf, gt_gdf, match_radius_m):
    """
    Match detections to ground-truth using geographic distance.
    Each GT can match at most one detection (highest conf wins).
    Each detection can match at most one GT.

    Returns:
        tp_det_idx:  set of detection indices that are true positives
        tp_gt_idx:   set of GT indices that were matched
        fn_gt_idx:   set of GT indices that were missed
        fp_det_idx:  set of detection indices that are false positives
    """
    if len(det_gdf) == 0:
        return set(), set(), set(range(len(gt_gdf))), set()
    if len(gt_gdf) == 0:
        return set(), set(), set(), set(range(len(det_gdf)))

    # Ensure same CRS
    if det_gdf.crs != gt_gdf.crs:
        gt_gdf = gt_gdf.to_crs(det_gdf.crs)

    # Build distance matrix (det x gt) in metres
    det_coords = np.array([(g.x, g.y) for g in det_gdf.geometry])
    gt_coords  = np.array([(g.x, g.y) for g in gt_gdf.geometry])

    # pairwise distances
    dx = det_coords[:, 0:1] - gt_coords[:, 0:1].T  # (n_det, n_gt)
    dy = det_coords[:, 1:2] - gt_coords[:, 1:2].T
    dist = np.sqrt(dx**2 + dy**2)

    # Greedy matching: iterate detections by descending confidence
    conf_order = det_gdf['confidence'].values.argsort()[::-1]
    tp_det_idx = set()
    tp_gt_idx  = set()

    for det_i in conf_order:
        # Find closest unmatched GT within radius
        dists_to_gt = dist[det_i]
        candidates = [
            (d, gt_j) for gt_j, d in enumerate(dists_to_gt)
            if d <= match_radius_m and gt_j not in tp_gt_idx
        ]
        if candidates:
            best_dist, best_gt = min(candidates, key=lambda x: x[0])
            tp_det_idx.add(det_i)
            tp_gt_idx.add(best_gt)

    fn_gt_idx  = set(range(len(gt_gdf))) - tp_gt_idx
    fp_det_idx = set(range(len(det_gdf))) - tp_det_idx

    return tp_det_idx, tp_gt_idx, fn_gt_idx, fp_det_idx


def compute_metrics_at_thresholds(det_gdf, gt_gdf, match_radius_m,
                                   thresholds=None):
    """
    Compute precision, recall, F1 at multiple confidence thresholds.
    Returns a DataFrame with one row per threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.95, 0.05)

    rows = []
    for thr in thresholds:
        filtered = det_gdf[det_gdf['confidence'] >= thr].reset_index(drop=True)
        tp, tp_gt, fn, fp = match_detections_to_gt(filtered, gt_gdf, match_radius_m)

        n_tp = len(tp)
        n_fp = len(fp)
        n_fn = len(fn)

        precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
        recall    = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        rows.append({
            'threshold': round(thr, 2),
            'TP': n_tp, 'FP': n_fp, 'FN': n_fn,
            'precision': round(precision, 3),
            'recall':    round(recall, 3),
            'F1':        round(f1, 3),
            'n_dets':    len(filtered),
        })

    return pd.DataFrame(rows)


def evaluate_label_set(label_path, label_name, all_detections_gdf,
                       image_paths, pixel_size, tumulus_diameter,
                       patch_size, model, output_dir,
                       conf_threshold, iou_threshold,
                       band_order, inference_combos,
                       match_radius_m):
    """
    Evaluate detections against a ground-truth label set.

    FIXED in v3:
    - Uses geographic distance matching (not patch-pixel coords)
    - Computes precision + recall + F1 at multiple thresholds
    - Exports false positives as GeoPackage for hard negative mining
    - Visualisation uses proper matching results
    """
    gt_gdf = gpd.read_file(label_path)
    if gt_gdf.crs != all_detections_gdf.crs:
        gt_gdf = gt_gdf.to_crs(all_detections_gdf.crs)

    print(f"\n{'='*60}")
    print(f"  EVALUATING: {label_name.upper()} SET ({len(gt_gdf)} ground-truth points)")
    print(f"  Match radius: {match_radius_m}m")
    print(f"{'='*60}")

    # ── Metrics at multiple thresholds ──────────────────────────────────────
    metrics_df = compute_metrics_at_thresholds(
        all_detections_gdf, gt_gdf, match_radius_m
    )
    print(f"\n  Confidence sweep:")
    print(metrics_df.to_string(index=False))

    # Find optimal F1 threshold
    best_row = metrics_df.loc[metrics_df['F1'].idxmax()]
    best_thr = best_row['threshold']
    print(f"\n  ★ Best F1={best_row['F1']:.3f} at conf≥{best_thr:.2f}"
          f"  (P={best_row['precision']:.3f}, R={best_row['recall']:.3f})")

    # Save metrics CSV
    csv_path = os.path.join(output_dir, f'metrics_{label_name.lower()}.csv')
    metrics_df.to_csv(csv_path, index=False)
    print(f"  Saved metrics: {csv_path}")

    # ── Match at the operating conf_threshold ──────────────────────────────
    det_filtered = all_detections_gdf[
        all_detections_gdf['confidence'] >= conf_threshold
    ].reset_index(drop=True)

    tp_det, tp_gt, fn_gt, fp_det = match_detections_to_gt(
        det_filtered, gt_gdf, match_radius_m
    )

    n_tp, n_fp, n_fn = len(tp_det), len(fp_det), len(fn_gt)
    prec = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    rec  = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  At operating threshold (conf≥{conf_threshold}):")
    print(f"    TP={n_tp}  FP={n_fp}  FN={n_fn}")
    print(f"    Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}")

    # ── Export false positives for hard negative mining ─────────────────────
    if n_fp > 0:
        fp_gdf = det_filtered.iloc[sorted(fp_det)].copy()
        fp_gdf = fp_gdf.reset_index(drop=True)
        fp_path = os.path.join(output_dir,
                               f'false_positives_{label_name.lower()}.gpkg')
        fp_gdf.to_file(fp_path, driver='GPKG', layer='false_positives')
        print(f"  ★ Exported {n_fp} false positives → {fp_path}")
        print(f"    Use these as hard negatives in training!")

    # ── Visualisation: per-GT-point patches ────────────────────────────────
    n = len(gt_gdf)
    ncols = min(4, max(1, n))
    nrows = max(1, (n + ncols - 1) // ncols)
    bbox_px = int(tumulus_diameter / pixel_size)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx in range(n):
        gt_row = gt_gdf.iloc[idx]
        gt_x, gt_y = gt_row.geometry.x, gt_row.geometry.y
        img_path = find_image_for_point(image_paths, gt_x, gt_y)

        if img_path is None:
            axes[idx].set_title(f"#{idx}: not in any image", color='gray')
            axes[idx].axis('off')
            continue

        with rasterio.open(img_path) as src:
            pr, pc = rowcol(src.transform, gt_x, gt_y)
            pr, pc = int(pr), int(pc)
            half = patch_size // 2
            r_off = int(np.clip(pr - half, 0, src.height - patch_size))
            c_off = int(np.clip(pc - half, 0, src.width - patch_size))
            window = Window(c_off, r_off, patch_size, patch_size)
            data = src.read(window=window)
            n_bands = src.count
            src_transform = src.transform

        rgb = normalize_to_uint8(data,
            (band_order['red'], band_order['green'], band_order['blue']))

        # GT position in patch pixel coords
        gt_px_x = pc - c_off
        gt_px_y = pr - r_off

        axes[idx].imshow(rgb)
        circle = plt.Circle((gt_px_x, gt_px_y), bbox_px // 2,
                            fill=False, color='lime', linewidth=2, linestyle='--')
        axes[idx].add_patch(circle)

        # Find nearby detections using GEOGRAPHIC distance
        is_detected = idx in tp_gt
        nearby_dets = []
        for _, det in det_filtered.iterrows():
            dx = det.geometry.x - gt_x
            dy = det.geometry.y - gt_y
            dist_m = np.sqrt(dx**2 + dy**2)
            if dist_m < patch_size * pixel_size / 2:
                # Convert detection geo coords to patch pixel coords
                det_pr, det_pc = rowcol(src_transform, det.geometry.x, det.geometry.y)
                det_px_x = int(det_pc) - c_off
                det_px_y = int(det_pr) - r_off
                nearby_dets.append((det_px_x, det_px_y, det['confidence'],
                                   det['diameter_m'], dist_m))

        for (dpx, dpy, conf, diam, dist_m) in nearby_dets:
            r_px = max(5, int(diam / pixel_size / 2))
            is_match = dist_m <= match_radius_m
            color = 'cyan' if is_match else 'red'
            circ = plt.Circle((dpx, dpy), r_px,
                             fill=False, color=color, linewidth=2)
            axes[idx].add_patch(circ)
            axes[idx].text(dpx + r_px + 2, dpy, f'{conf:.2f}',
                          color=color, fontsize=8, fontweight='bold',
                          backgroundcolor='white')

        status = "✓ TP" if is_detected else "✗ FN"
        color  = 'green' if is_detected else 'red'
        axes[idx].set_title(f"#{idx}: {status}", color=color, fontsize=10)
        axes[idx].axis('off')

    for idx in range(n, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(
        f"{label_name} Set — TP={n_tp}, FN={n_fn}, FP={n_fp} | "
        f"P={prec:.2f}  R={rec:.2f}  F1={f1:.2f}\n"
        f"conf≥{conf_threshold} | match_radius={match_radius_m}m | "
        f"Green dashed=GT | Cyan=TP det | Red=FP det",
        fontsize=11, y=1.01
    )
    plt.tight_layout()
    out_path = os.path.join(output_dir, f'evaluation_{label_name.lower()}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualisation: {out_path}")

    # ── P-R curve plot ─────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(metrics_df['threshold'], metrics_df['precision'],
             'b-o', markersize=4, label='Precision')
    ax1.plot(metrics_df['threshold'], metrics_df['recall'],
             'r-o', markersize=4, label='Recall')
    ax1.plot(metrics_df['threshold'], metrics_df['F1'],
             'g-s', markersize=5, label='F1', linewidth=2)
    ax1.axvline(best_thr, color='green', linestyle=':', alpha=0.7,
                label=f'Best F1 @ {best_thr:.2f}')
    ax1.axvline(conf_threshold, color='gray', linestyle='--', alpha=0.5,
                label=f'Operating @ {conf_threshold:.2f}')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{label_name}: Precision / Recall / F1 vs Confidence')
    ax1.legend(fontsize=8)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2.plot(metrics_df['recall'], metrics_df['precision'],
             'k-o', markersize=4)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'{label_name}: Precision-Recall Curve')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pr_path = os.path.join(output_dir, f'pr_curve_{label_name.lower()}.png')
    plt.savefig(pr_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved P-R curve: {pr_path}")

    return {
        'label_name': label_name,
        'n_gt': len(gt_gdf),
        'n_det': len(det_filtered),
        'TP': n_tp, 'FP': n_fp, 'FN': n_fn,
        'precision': prec, 'recall': rec, 'F1': f1,
        'best_threshold': best_thr,
        'best_F1': best_row['F1'],
        'metrics_df': metrics_df,
    }


# ── Visualisations ───────────────────────────────────────────────────────────────

def visualize_top_detections(gdf, image_paths, pixel_size,
                             tumulus_diameter, patch_size, output_dir, n=16):
    top = gdf.nlargest(min(n, len(gdf)), 'confidence')
    ncols = 4
    nrows = max(1, (len(top) + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
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
            view = 300
            r_off = int(np.clip(pr - view//2, 0, src.height - view))
            c_off = int(np.clip(pc - view//2, 0, src.width  - view))
            window = Window(c_off, r_off, view, view)
            data = src.read(window=window)

        rgb = normalize_to_uint8(data,
            (BAND_ORDER['red'], BAND_ORDER['green'], BAND_ORDER['blue']))
        cx, cy = pc - c_off, pr - r_off

        axes[idx].imshow(rgb)
        r_px = int(det['diameter_m'] / pixel_size / 2)
        circle = plt.Circle((cx, cy), r_px, fill=False, color='red', linewidth=2)
        axes[idx].add_patch(circle)
        axes[idx].set_title(f"conf={det['confidence']:.2f}  d≈{det['diameter_m']:.0f}m",
                           fontsize=9)
        axes[idx].axis('off')

    for idx in range(len(top), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Top {len(top)} Detections by Confidence", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, 'top_detections.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_confidence_distribution(gdf, tumulus_diameter, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(gdf['confidence'], bins=20, edgecolor='black', color='steelblue')
    ax1.axvline(0.25, color='orange', linestyle='--', label='conf=0.25')
    ax1.axvline(0.50, color='red',    linestyle='--', label='conf=0.50')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Detection Confidence Distribution')
    ax1.legend()

    ax2.hist(gdf['diameter_m'], bins=20, edgecolor='black', color='darkorange')
    ax2.axvline(tumulus_diameter, color='red', linestyle='--',
                label=f'Expected ({tumulus_diameter}m)')
    ax2.set_xlabel('Estimated Diameter (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Detected Object Size Distribution')
    ax2.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, 'detection_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ── Hard Negative Export Utility ─────────────────────────────────────────────────

def export_hard_negatives(all_detections_gdf, gt_gdfs, match_radius_m,
                          conf_threshold_min, conf_threshold_max,
                          output_dir):
    """
    Export false positive detections as hard negatives for retraining.

    Merges all GT sets to identify which detections are FPs across
    ALL known ground truth, then filters by confidence range.

    Args:
        all_detections_gdf: full detection GeoDataFrame
        gt_gdfs: list of ground-truth GeoDataFrames (train+val+test)
        match_radius_m: matching radius in metres
        conf_threshold_min: minimum confidence for hard negatives
        conf_threshold_max: maximum confidence for hard negatives
        output_dir: output directory
    """
    # Merge all GT
    all_gt = pd.concat(gt_gdfs, ignore_index=True)
    all_gt = gpd.GeoDataFrame(all_gt, crs=gt_gdfs[0].crs)

    # Filter detections by confidence range
    mask = ((all_detections_gdf['confidence'] >= conf_threshold_min) &
            (all_detections_gdf['confidence'] <= conf_threshold_max))
    det_filtered = all_detections_gdf[mask].reset_index(drop=True)

    # Match against all GT
    tp_det, _, _, fp_det = match_detections_to_gt(
        det_filtered, all_gt, match_radius_m
    )

    if len(fp_det) == 0:
        print("  No false positives found in the specified confidence range.")
        return None

    fp_gdf = det_filtered.iloc[sorted(fp_det)].copy().reset_index(drop=True)

    # Add a 'tier' column for priority
    fp_gdf['tier'] = pd.cut(
        fp_gdf['confidence'],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['low', 'medium', 'high', 'very_high']
    )

    fp_path = os.path.join(output_dir, 'hard_negatives_all.gpkg')
    fp_gdf.to_file(fp_path, driver='GPKG', layer='hard_negatives')

    print(f"\n  ★ HARD NEGATIVES EXPORT")
    print(f"    Total FPs (conf {conf_threshold_min:.2f}–{conf_threshold_max:.2f}): "
          f"{len(fp_gdf)}")
    print(f"    Tier breakdown:")
    print(f"      very_high (>0.7): {(fp_gdf['tier'] == 'very_high').sum()}")
    print(f"      high (0.5–0.7):   {(fp_gdf['tier'] == 'high').sum()}")
    print(f"      medium (0.3–0.5): {(fp_gdf['tier'] == 'medium').sum()}")
    print(f"      low (0.15–0.3):   {(fp_gdf['tier'] == 'low').sum()}")
    print(f"    Saved: {fp_path}")
    print(f"    → Use 'high' and 'very_high' tiers first for maximum impact")

    return fp_gdf


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    from ultralytics import YOLO

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        sys.exit(1)
    missing = [p for p in IMAGE_PATHS if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Missing images: {missing}")
        sys.exit(1)

    print("=" * 60)
    print("  TUMULUS DETECTION  [v3]")
    print("=" * 60)
    print(f"  Model:         {MODEL_PATH}")
    print(f"  Images:        {len(IMAGE_PATHS)}")
    print(f"  Combos:        {INFERENCE_COMBOS}")
    print(f"  Conf:          {CONF_THRESHOLD}  |  TTA: {'on' if USE_TTA else 'off'}")
    print(f"  Match radius:  {MATCH_RADIUS_M}m")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    all_detections, all_crs = [], None

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
            inference_combos=INFERENCE_COMBOS,
            band_order=BAND_ORDER,
            nodata_skip_threshold=NODATA_SKIP_THRESHOLD,
        )
        all_detections.extend(dets)
        if all_crs is None:
            all_crs = crs

    if not all_detections:
        print("\nNo detections. Try lowering CONF_THRESHOLD.")
        return

    gdf = gpd.GeoDataFrame(all_detections, crs=all_crs)
    gdf = gdf.drop(columns=['row', 'col'])
    gdf['id'] = range(1, len(gdf) + 1)

    # Save full results
    gpkg_path = os.path.join(OUTPUT_DIR, 'tumulus_detections.gpkg')
    gdf.to_file(gpkg_path, driver='GPKG', layer='detections')

    # Save confidence-filtered subsets
    for threshold in EXPORT_CONF_THRESHOLDS:
        subset = gdf[gdf['confidence'] >= threshold]
        if len(subset) > 0:
            subset.to_file(
                os.path.join(OUTPUT_DIR, f'detections_conf{int(threshold*100)}.gpkg'),
                driver='GPKG', layer='detections'
            )

    print(f"\n{'='*60}")
    print(f"  TOTAL DETECTIONS: {len(gdf)}")
    print(f"  Saved to: {gpkg_path}")
    print(f"{'='*60}")
    print(gdf[['confidence', 'diameter_m', 'source']].describe())

    # ── Evaluate against all label sets ────────────────────────────────────
    eval_results = []

    label_sets = [
        (TEST_LABELS_PATH,  "Test"),
        (VAL_LABELS_PATH,   "Validation"),
        (TRAIN_LABELS_PATH, "Training"),
    ]

    for label_path, label_name in label_sets:
        if label_path and os.path.exists(label_path):
            result = evaluate_label_set(
                label_path, label_name, gdf,
                IMAGE_PATHS, PIXEL_SIZE, TUMULUS_DIAMETER, PATCH_SIZE,
                model, OUTPUT_DIR, CONF_THRESHOLD, IOU_THRESHOLD,
                BAND_ORDER, INFERENCE_COMBOS, MATCH_RADIUS_M,
            )
            eval_results.append(result)

    # ── Summary table ──────────────────────────────────────────────────────
    if eval_results:
        print(f"\n{'='*60}")
        print(f"  EVALUATION SUMMARY (conf≥{CONF_THRESHOLD})")
        print(f"{'='*60}")
        print(f"  {'Set':<12} {'GT':>4} {'Det':>5} {'TP':>4} {'FP':>5} "
              f"{'FN':>4} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Best@':>7}")
        print(f"  {'-'*58}")
        for r in eval_results:
            print(f"  {r['label_name']:<12} {r['n_gt']:>4} {r['n_det']:>5} "
                  f"{r['TP']:>4} {r['FP']:>5} {r['FN']:>4} "
                  f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['F1']:>6.3f} "
                  f"{r['best_threshold']:>5.2f}={r['best_F1']:.2f}")
        print(f"{'='*60}")

    # ── Export hard negatives across all GT ─────────────────────────────────
    gt_gdfs = []
    for label_path, _ in label_sets:
        if label_path and os.path.exists(label_path):
            g = gpd.read_file(label_path)
            if g.crs != gdf.crs:
                g = g.to_crs(gdf.crs)
            gt_gdfs.append(g)

    if gt_gdfs:
        export_hard_negatives(
            gdf, gt_gdfs, MATCH_RADIUS_M,
            conf_threshold_min=0.40,  # Focus on confident FPs
            conf_threshold_max=1.0,
            output_dir=OUTPUT_DIR,
        )

    # ── Standard visualisations ────────────────────────────────────────────
    visualize_top_detections(gdf, IMAGE_PATHS, PIXEL_SIZE,
                             TUMULUS_DIAMETER, PATCH_SIZE, OUTPUT_DIR)
    plot_confidence_distribution(gdf, TUMULUS_DIAMETER, OUTPUT_DIR)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}/")
    print("Load tumulus_detections.gpkg in QGIS for full spatial review.")
    print("Load hard_negatives_all.gpkg for FP locations to add as hard negatives.")


if __name__ == "__main__":
    main()