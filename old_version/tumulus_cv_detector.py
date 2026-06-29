"""
Tumulus Detector v2 — Radial Symmetry Transform
================================================
Detects circular stone structures by finding points where image gradients
converge radially, regardless of whether the structure is brighter or darker
than its surroundings.

Why this works better than texture variance for tumuli:
    - Tumuli are circular arrangements of stones, not high-contrast blobs
    - The radial symmetry transform detects ANY circular structure by finding
      where gradient vectors point toward a common center
    - Works on structures with subtle tonal differences from background
    - Naturally multi-scale (searches across a range of radii)

Based on: Loy & Zelinsky (2003) "Fast Radial Symmetry for Detecting
Points of Interest", IEEE TPAMI 25(8).

Usage:
    python tumulus_detector_v2.py -i pleiades.tif -o candidates.gpkg \
        --min_diameter 8 --max_diameter 25 --workers -1

Requirements:
    pip install rasterio numpy scipy scikit-image geopandas shapely opencv-python

Author: Adapted for pastoral archaeology in Dhofar, Oman (CAMP project)
"""

import argparse
import time
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import xy
from scipy.ndimage import gaussian_filter, uniform_filter
from skimage.feature import peak_local_max
from skimage.measure import label as sk_label, regionprops_table
from skimage.morphology import disk
import cv2
import geopandas as gpd
from shapely.geometry import Point
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CORE: FAST RADIAL SYMMETRY TRANSFORM
# ============================================================

def fast_radial_symmetry(image, radii, alpha=2.0, mode='both'):
    """
    Fast Radial Symmetry Transform (Loy & Zelinsky 2003).

    For each pixel, compute the gradient and "vote" for a center point
    at distance r along the gradient direction. Accumulate votes across
    all radii. Peaks in the accumulator = centers of circular structures.

    Parameters
    ----------
    image : np.ndarray (float64)
        Grayscale image, normalized 0-1.
    radii : list of int
        Radii to search (in pixels).
    alpha : float
        Radial strictness parameter. Higher = stricter radial symmetry.
        Use 2.0 for general circular features.
    mode : str
        'bright' = detect bright circles on dark background
        'dark' = detect dark circles on bright background
        'both' = detect both (recommended for tumuli)

    Returns
    -------
    S : np.ndarray
        Symmetry response map. Peaks = circle centers.
    """
    h, w = image.shape

    # Compute gradients
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag = np.maximum(mag, 1e-10)  # Avoid division by zero

    # Normalize gradient directions
    gx_norm = gx / mag
    gy_norm = gy / mag

    S = np.zeros((h, w), dtype=np.float64)

    for r in radii:
        # Orientation projection image (votes) and magnitude image
        On = np.zeros((h, w), dtype=np.float64)
        Mn = np.zeros((h, w), dtype=np.float64)

        # For each pixel with significant gradient, cast a vote
        # at the positively-affected pixel (p + r * g_hat)
        # and negatively-affected pixel (p - r * g_hat)

        # Threshold: only use pixels with meaningful gradient
        grad_threshold = np.percentile(mag[mag > 0], 50)
        significant = mag > grad_threshold

        ys, xs = np.where(significant)
        if len(ys) == 0:
            continue

        gx_s = gx_norm[ys, xs]
        gy_s = gy_norm[ys, xs]
        mag_s = mag[ys, xs]

        # Positive affected pixels (gradient points toward center)
        px_pos = (xs + np.round(r * gx_s)).astype(int)
        py_pos = (ys + np.round(r * gy_s)).astype(int)

        # Negative affected pixels (gradient points away from center)
        px_neg = (xs - np.round(r * gx_s)).astype(int)
        py_neg = (ys - np.round(r * gy_s)).astype(int)

        # Clip to image bounds
        valid_pos = (px_pos >= 0) & (px_pos < w) & (py_pos >= 0) & (py_pos < h)
        valid_neg = (px_neg >= 0) & (px_neg < w) & (py_neg >= 0) & (py_neg < h)

        if mode in ('bright', 'both'):
            # Bright circles: gradients point outward from center
            np.add.at(On, (py_pos[valid_pos], px_pos[valid_pos]), 1)
            np.add.at(Mn, (py_pos[valid_pos], px_pos[valid_pos]), mag_s[valid_pos])

        if mode in ('dark', 'both'):
            # Dark circles: gradients point inward toward center
            np.add.at(On, (py_neg[valid_neg], px_neg[valid_neg]), 1)
            np.add.at(Mn, (py_neg[valid_neg], px_neg[valid_neg]), mag_s[valid_neg])

        # Clamp On to reasonable range
        On_max = max(np.abs(On).max(), 1)
        kappa = min(On_max, r)

        # Compute symmetry contribution for this radius
        # F_n = Mn/kappa * (|On|/kappa)^alpha
        Fn = (Mn / (kappa + 1e-10)) * np.power(np.abs(On) / (kappa + 1e-10), alpha)

        # Smooth with Gaussian proportional to radius
        sigma = max(1.0, r * 0.25)
        Fn = gaussian_filter(Fn, sigma=sigma)

        S += Fn

    # Normalize
    if S.max() > 0:
        S = S / S.max()

    return S


def multi_scale_radial_symmetry(image, min_radius, max_radius, n_scales=12,
                                 alpha=2.0, mode='both'):
    """
    Multi-scale radial symmetry with scale-space response tracking.

    Returns both the combined symmetry map and per-scale responses
    for determining the best-fit radius of each detection.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (0-1).
    min_radius, max_radius : int
        Radius search range in pixels.
    n_scales : int
        Number of radius scales to evaluate.
    alpha : float
        Radial strictness.
    mode : str
        'bright', 'dark', or 'both'.

    Returns
    -------
    S_combined : np.ndarray
        Combined symmetry response across all scales.
    scale_responses : list of (radius, response_map)
        Per-scale response maps for radius estimation.
    """
    radii = np.linspace(min_radius, max_radius, n_scales).astype(int)
    radii = np.unique(radii)  # Remove duplicates

    h, w = image.shape

    # Compute gradients once
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    mag_safe = np.maximum(mag, 1e-10)
    gx_norm = gx / mag_safe
    gy_norm = gy / mag_safe

    grad_threshold = np.percentile(mag[mag > 0], 50) if np.any(mag > 0) else 0
    significant = mag > grad_threshold
    ys, xs = np.where(significant)

    if len(ys) == 0:
        return np.zeros((h, w)), []

    gx_s = gx_norm[ys, xs]
    gy_s = gy_norm[ys, xs]
    mag_s = mag[ys, xs]

    S_combined = np.zeros((h, w), dtype=np.float64)
    scale_responses = []

    for r in radii:
        On = np.zeros((h, w), dtype=np.float64)
        Mn = np.zeros((h, w), dtype=np.float64)

        # Positive votes
        px_pos = (xs + np.round(r * gx_s)).astype(int)
        py_pos = (ys + np.round(r * gy_s)).astype(int)
        px_neg = (xs - np.round(r * gx_s)).astype(int)
        py_neg = (ys - np.round(r * gy_s)).astype(int)

        valid_pos = (px_pos >= 0) & (px_pos < w) & (py_pos >= 0) & (py_pos < h)
        valid_neg = (px_neg >= 0) & (px_neg < w) & (py_neg >= 0) & (py_neg < h)

        if mode in ('bright', 'both'):
            np.add.at(On, (py_pos[valid_pos], px_pos[valid_pos]), 1)
            np.add.at(Mn, (py_pos[valid_pos], px_pos[valid_pos]), mag_s[valid_pos])

        if mode in ('dark', 'both'):
            np.add.at(On, (py_neg[valid_neg], px_neg[valid_neg]), 1)
            np.add.at(Mn, (py_neg[valid_neg], px_neg[valid_neg]), mag_s[valid_neg])

        On_max = max(np.abs(On).max(), 1)
        kappa = min(On_max, r)

        Fn = (Mn / (kappa + 1e-10)) * np.power(np.abs(On) / (kappa + 1e-10), alpha)

        sigma = max(1.0, r * 0.25)
        Fn = gaussian_filter(Fn, sigma=sigma)

        if Fn.max() > 0:
            Fn_norm = Fn / Fn.max()
        else:
            Fn_norm = Fn

        scale_responses.append((int(r), Fn_norm))
        S_combined += Fn_norm

    if S_combined.max() > 0:
        S_combined /= S_combined.max()

    return S_combined, scale_responses


# ============================================================
# TILE PROCESSING (top-level for multiprocessing)
# ============================================================

def process_single_tile(args):
    """Process one tile and return candidates in full-image coordinates."""
    image_path, col_off, row_off, win_width, win_height, config = args

    min_radius_px = config['min_radius_px']
    max_radius_px = config['max_radius_px']
    min_diameter_m = config['min_diameter_m']
    max_diameter_m = config['max_diameter_m']
    pixel_size = config['pixel_size']
    tile_id = config.get('tile_id', '')
    min_peak_distance = config.get('min_peak_distance', min_radius_px)

    t0 = time.time()

    # Read tile
    with rasterio.open(image_path) as src:
        window = Window(col_off, row_off, win_width, win_height)
        bands = src.read(window=window).astype(np.float32)

    nodata_mask = bands[0] == 0
    if nodata_mask.sum() / nodata_mask.size > 0.9:
        return []

    # Composite grayscale (emphasize visible bands for stone contrast)
    R = bands[0].astype(np.float64)
    G = bands[1].astype(np.float64)
    B = bands[2].astype(np.float64)
    NIR = bands[3].astype(np.float64)
    composite = 0.30 * R + 0.30 * G + 0.25 * B + 0.15 * NIR
    del R, G, B, NIR, bands

    valid = composite[~nodata_mask]
    if len(valid) == 0:
        return []
    p2, p98 = np.percentile(valid, [2, 98])
    composite = np.clip((composite - p2) / (p98 - p2 + 1e-10), 0, 1)
    composite[nodata_mask] = 0

    # --- Optional: light pre-smoothing to reduce noise from individual rocks ---
    # This helps the gradient computation focus on the meso-scale structure
    # (the tumulus outline) rather than micro-scale texture (individual stones)
    pre_smooth = gaussian_filter(composite, sigma=2.0)

    # --- Multi-scale radial symmetry transform ---
    S_combined, scale_responses = multi_scale_radial_symmetry(
        pre_smooth,
        min_radius=min_radius_px,
        max_radius=max_radius_px,
        n_scales=10,
        alpha=2.0,
        mode='both',
    )

    # Mask nodata
    S_combined[nodata_mask] = 0

    # --- Detect peaks in symmetry response ---
    # Adaptive threshold: peaks must be significantly above local background
    if S_combined.max() == 0:
        return []

    threshold = max(0.15, np.percentile(S_combined[S_combined > 0], 90))

    peaks = peak_local_max(
        S_combined,
        min_distance=min_peak_distance,
        threshold_abs=threshold,
        exclude_border=max_radius_px,
    )

    if len(peaks) == 0:
        return []

    # --- For each peak, determine best radius from scale responses ---
    candidates = []
    for peak in peaks:
        row, col = int(peak[0]), int(peak[1])

        # Find which scale had the strongest response at this location
        best_radius = min_radius_px
        best_response = 0
        for radius, response_map in scale_responses:
            val = response_map[row, col]
            if val > best_response:
                best_response = val
                best_radius = radius

        # Compute local properties for confidence scoring
        symmetry_strength = S_combined[row, col]

        # Check circularity using the symmetry response profile
        # A true circle should have similar symmetry in all directions
        r = best_radius
        angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        ring_values = []
        for angle in angles:
            ry = int(row + r * 0.7 * np.sin(angle))
            rx = int(col + r * 0.7 * np.cos(angle))
            if 0 <= ry < S_combined.shape[0] and 0 <= rx < S_combined.shape[1]:
                ring_values.append(composite[ry, rx])

        if len(ring_values) >= 8:
            ring_std = np.std(ring_values)
            ring_mean = np.mean(ring_values)
            # Low std relative to mean = consistent circular structure
            ring_uniformity = 1.0 - min(1.0, ring_std / (ring_mean + 1e-10))
        else:
            ring_uniformity = 0.5

        # Confidence score
        diameter_m = best_radius * 2 * pixel_size
        size_score = max(0, 1.0 - abs(diameter_m - 15) / 15)

        confidence = (
            0.5 * symmetry_strength +
            0.25 * ring_uniformity +
            0.25 * size_score
        )

        candidates.append({
            'row': row + row_off,
            'col': col + col_off,
            'radius_px': best_radius,
            'radius_m': best_radius * pixel_size,
            'diameter_m': diameter_m,
            'symmetry_strength': float(symmetry_strength),
            'ring_uniformity': float(ring_uniformity),
            'confidence': float(np.clip(confidence, 0, 1)),
        })

    elapsed = time.time() - t0
    print(f"  Tile {tile_id}: {len(peaks)} peaks -> {len(candidates)} candidates ({elapsed:.1f}s)")

    return candidates


def _remove_duplicates_fast(candidates, min_dist_px=67):
    """Vectorized duplicate removal keeping highest confidence."""
    if len(candidates) <= 1:
        return candidates

    candidates.sort(key=lambda x: x['confidence'], reverse=True)
    coords = np.array([(c['row'], c['col']) for c in candidates])
    keep_mask = np.ones(len(candidates), dtype=bool)

    for i in range(len(candidates)):
        if not keep_mask[i]:
            continue
        if i + 1 < len(candidates):
            diffs = coords[i + 1:] - coords[i]
            dists = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)
            too_close = np.where(dists < min_dist_px)[0] + i + 1
            keep_mask[too_close] = False

    return [c for c, k in zip(candidates, keep_mask) if k]


# ============================================================
# MAIN DETECTOR
# ============================================================

class TumulusDetectorV2:
    """
    Detect stone tumuli using radial symmetry transform.
    Finds circular structures by gradient convergence, not texture intensity.
    """

    def __init__(self, min_diameter_m=8, max_diameter_m=25, pixel_size=0.15):
        self.min_diameter_m = min_diameter_m
        self.max_diameter_m = max_diameter_m
        self.pixel_size = pixel_size
        self.min_radius_px = int((min_diameter_m / 2) / pixel_size)
        self.max_radius_px = int((max_diameter_m / 2) / pixel_size)

        print(f"=== Tumulus Detector v2 (Radial Symmetry) ===")
        print(f"Size range: {min_diameter_m}-{max_diameter_m}m")
        print(f"Pixel size: {pixel_size}m")
        print(f"Radius range: {self.min_radius_px}-{self.max_radius_px} px")

    def run(self, image_path, output_path, tile_size=2000, overlap=300, workers=4):
        """
        Run detection on full image with parallel tile processing.

        Parameters
        ----------
        image_path : str
            Path to the Pleiades GeoTIFF.
        output_path : str
            Output GeoPackage path.
        tile_size : int
            Tile size in pixels. Smaller tiles = less memory, more tiles.
            2000 recommended for radial symmetry (more compute-intensive).
        overlap : int
            Overlap between tiles. Should be >= max_radius_px to avoid
            missing features at tile boundaries.
        workers : int
            Number of parallel workers (-1 for all cores).
        """
        t_start = time.time()

        with rasterio.open(image_path) as src:
            height = src.height
            width = src.width
            crs = src.crs
            full_transform = src.transform

        # Ensure overlap is at least the max search radius
        overlap = max(overlap, self.max_radius_px + 10)

        print(f"\nImage: {width} x {height} px")
        print(f"Tiles: {tile_size}x{tile_size}, overlap: {overlap}px")
        print(f"Workers: {workers}\n")

        config = {
            'min_radius_px': self.min_radius_px,
            'max_radius_px': self.max_radius_px,
            'min_diameter_m': self.min_diameter_m,
            'max_diameter_m': self.max_diameter_m,
            'pixel_size': self.pixel_size,
            'min_peak_distance': self.min_radius_px,
        }

        stride = tile_size - overlap
        tile_args = []
        tile_idx = 0

        for ty in range(0, height, stride):
            for tx in range(0, width, stride):
                win_h = min(tile_size, height - ty)
                win_w = min(tile_size, width - tx)
                if win_h < self.max_radius_px * 3 or win_w < self.max_radius_px * 3:
                    continue
                tc = dict(config)
                tc['tile_id'] = str(tile_idx)
                tile_args.append((image_path, tx, ty, win_w, win_h, tc))
                tile_idx += 1

        print(f"Total tiles: {len(tile_args)}")

        # Parallel processing
        all_candidates = []
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(process_single_tile, a): i
                    for i, a in enumerate(tile_args)
                }
                for future in as_completed(futures):
                    try:
                        all_candidates.extend(future.result())
                    except Exception as e:
                        print(f"  Tile error: {e}")
        else:
            for a in tile_args:
                all_candidates.extend(process_single_tile(a))

        # Deduplicate across tile boundaries
        print(f"\nPre-dedup: {len(all_candidates)}")
        all_candidates = _remove_duplicates_fast(
            all_candidates,
            min_dist_px=self.min_diameter_m * 0.75 / self.pixel_size
        )
        print(f"Final: {len(all_candidates)}")

        if not all_candidates:
            print("No candidates found.")
            return gpd.GeoDataFrame()

        # Build GeoDataFrame
        features = []
        for i, c in enumerate(all_candidates):
            x, y = xy(full_transform, c['row'], c['col'])
            features.append({
                'geometry': Point(x, y),
                'id': i + 1,
                'x': x, 'y': y,
                'radius_m': round(c['radius_m'], 2),
                'diameter_m': round(c['diameter_m'], 2),
                'symmetry': round(c['symmetry_strength'], 3),
                'uniformity': round(c['ring_uniformity'], 3),
                'confidence': round(c['confidence'], 3),
                'row': c['row'],
                'col': c['col'],
            })

        gdf = gpd.GeoDataFrame(features, crs=crs)

        # Save points
        gdf.to_file(output_path, driver='GPKG', layer='tumulus_candidates_points')

        # Save circles
        gdf_c = gdf.copy()
        gdf_c['geometry'] = gdf_c.apply(
            lambda r: Point(r['x'], r['y']).buffer(r['radius_m']), axis=1
        )
        gdf_c.to_file(output_path, driver='GPKG', layer='tumulus_candidates_circles')

        elapsed = time.time() - t_start
        print(f"\nSaved to: {output_path}")
        print(f"  Points: {len(gdf)} features")
        print(f"  Time: {elapsed:.0f}s ({elapsed / 60:.1f}min)")

        if len(gdf) > 0:
            print(f"\n--- Summary ---")
            print(f"  Diameter: {gdf['diameter_m'].min():.1f}-{gdf['diameter_m'].max():.1f}m")
            print(f"  Symmetry: {gdf['symmetry'].min():.3f}-{gdf['symmetry'].max():.3f}")
            print(f"  Confidence: {gdf['confidence'].min():.3f}-{gdf['confidence'].max():.3f}")
            print(f"  High conf (>0.6): {len(gdf[gdf['confidence'] > 0.6])}")
            print(f"  Med conf (0.4-0.6): {len(gdf[(gdf['confidence'] >= 0.4) & (gdf['confidence'] <= 0.6)])}")

        return gdf


def main():
    parser = argparse.ArgumentParser(
        description='Detect stone tumuli using radial symmetry transform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tumulus_detector_v2.py -i pleiades.tif -o candidates.gpkg --workers -1

  # Adjust for larger tumuli
  python tumulus_detector_v2.py -i pleiades.tif -o candidates.gpkg \\
      --min_diameter 10 --max_diameter 30

  # Smaller tiles if running out of memory
  python tumulus_detector_v2.py -i pleiades.tif -o candidates.gpkg --tile_size 1500
        """
    )
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--min_diameter', type=float, default=8)
    parser.add_argument('--max_diameter', type=float, default=25)
    parser.add_argument('--pixel_size', type=float, default=0.15)
    parser.add_argument('--tile_size', type=int, default=2000)
    parser.add_argument('--overlap', type=int, default=300)
    parser.add_argument('--workers', '-w', type=int, default=4)

    args = parser.parse_args()

    if args.workers == -1:
        import os
        args.workers = os.cpu_count()
        print(f"Using all {args.workers} cores")

    detector = TumulusDetectorV2(
        min_diameter_m=args.min_diameter,
        max_diameter_m=args.max_diameter,
        pixel_size=args.pixel_size,
    )

    detector.run(
        image_path=args.input,
        output_path=args.output,
        tile_size=args.tile_size,
        overlap=args.overlap,
        workers=args.workers,
    )


if __name__ == '__main__':
    main()