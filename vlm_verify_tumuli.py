"""
VLM Verification Pipeline for Tumulus Detection (v3)
=====================================================
KEY CHANGE in v3: Supports MULTIPLE Pleiades images via --image flag.
Points are automatically matched to the correct image based on bounds.

Usage:
    # Extract patches (multi-image support)
    python vlm_verify_tumuli_v3.py --extract_only \
        --detections ./output/detections/tumulus_detections-pleiades1.gpkg \
        --image ./data/pleiades1.TIF ./data/pleiades3.TIF \
        --labels ./tombs_training_clean.gpkg \
        --labels_test ./tombs_testing.gpkg \
        --backgrounds ./output/detections/backgrounds.gpkg \
        --output ./output/patches_pleiades1 \
        --patch_size 256

Author: Anton Koutoupas / Claude (Anthropic)
Project: Tumulus Detection - Somalia
"""

import argparse
import base64
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from PIL import Image
import requests
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

SYSTEM_PROMPT = """You are an expert archaeological remote sensing analyst specializing in detecting ancient stone tumuli (burial cairns/mounds) from very high resolution satellite imagery (Pleiades Neo, 15cm pixel size) in arid/semi-arid landscapes of the Horn of Africa (Somalia).

## What a tumulus looks like in this imagery:

1. **Circular or sub-circular cluster** of stones, typically 8-20 meters in diameter
2. **Dense, granular texture** — accumulated stones (10-50cm each) create a speckled bright/dark pattern
3. **Slightly brighter or differently-textured** compared to surrounding flat terrain
4. May show a **subtle shadow** on one side indicating slight elevation above ground
5. **Distinct boundary** (sharp or diffuse) with surrounding terrain
6. Typically **isolated** or in small groups, NOT continuous patterns

## Common false positives to reject:

- **Rocky outcrops**: Irregular/elongated shapes, connected to geological formations, linear fractures
- **Vegetation/bushes**: Dark patches, smaller (2-5m), irregular fuzzy edges — VERY common false positive
- **Wadi/drainage**: Linear or branching patterns
- **Sand/bare soil**: Uniform texture, no stone accumulation
- **Natural stone scatter**: Random distribution WITHOUT circular concentration
- **Shadow artifacts**: Dark patches without stone texture

## IMPORTANT: Be critical and skeptical. Many CNN detections are false positives. Only classify as TUMULUS if you see clear circular stone accumulation. When in doubt, classify as POSSIBLE_TUMULUS or NOT_TUMULUS.

Always respond with ONLY valid JSON."""

CLASSIFICATION_PROMPT = """Examine this satellite image patch (15cm/pixel, Pleiades Neo, natural color RGB) from an arid landscape in Somalia. A CNN detector flagged the feature near the CENTER of this patch as a potential ancient stone tumulus.

Be CRITICAL — many detections are false positives (vegetation, rock outcrops, shadows).

Classify the central feature. Respond with ONLY this JSON:
{"classification": "TUMULUS" or "POSSIBLE_TUMULUS" or "NOT_TUMULUS", "confidence": 0.0-1.0, "reasoning": "what you observe", "likely_feature": "tumulus" or "rock_outcrop" or "vegetation" or "sand" or "shadow" or "drainage" or "other"}"""


# ============================================================
# MULTI-IMAGE SUPPORT
# ============================================================

class ImageManager:
    """
    Manages multiple Pleiades images. For any given point (x, y),
    finds which image contains it and extracts the patch from there.
    Similar to find_image_for_point() in tumulus_yolo_v3.py.
    """
    
    def __init__(self, image_paths):
        self.images = []
        for path in image_paths:
            if not os.path.exists(path):
                print(f"  Warning: Image not found: {path}")
                continue
            with rasterio.open(path) as src:
                self.images.append({
                    "path": path,
                    "bounds": src.bounds,
                    "crs": src.crs,
                    "height": src.height,
                    "width": src.width,
                })
        
        if not self.images:
            raise ValueError("No valid images found!")
        
        print(f"  Loaded {len(self.images)} image(s):")
        for img in self.images:
            b = img["bounds"]
            print(f"    {Path(img['path']).name}: "
                  f"({b.left:.0f}, {b.bottom:.0f}) to ({b.right:.0f}, {b.top:.0f}) "
                  f"[{img['width']}x{img['height']}px]")
    
    @property
    def crs(self):
        return self.images[0]["crs"]
    
    def find_image_for_point(self, x, y):
        """Return the image path that contains point (x, y), or None."""
        for img in self.images:
            b = img["bounds"]
            if b.left <= x <= b.right and b.bottom <= y <= b.top:
                return img["path"]
        return None
    
    def extract_patch(self, x, y, patch_size=256):
        """
        Extract an RGB patch centered on (x, y) from whichever image contains it.
        Returns numpy array (H, W, 3) uint8, or None.
        """
        img_path = self.find_image_for_point(x, y)
        if img_path is None:
            return None
        
        with rasterio.open(img_path) as src:
            row, col = rowcol(src.transform, x, y)
            half = patch_size // 2
            
            row_start = int(row - half)
            col_start = int(col - half)
            
            # Reject edge cases — don't clamp!
            if (row_start < 0 or col_start < 0 or
                row_start + patch_size > src.height or
                col_start + patch_size > src.width):
                return None
            
            window = Window(col_start, row_start, patch_size, patch_size)
            
            try:
                data = src.read([1, 2, 3], window=window)  # (3, H, W)
                data = np.transpose(data, (1, 2, 0))       # (H, W, 3)
                
                if data.dtype != np.uint8:
                    p2, p98 = np.percentile(data, (2, 98))
                    if p98 > p2:
                        data = np.clip((data.astype(float) - p2) / (p98 - p2) * 255,
                                       0, 255).astype(np.uint8)
                    else:
                        data = np.zeros_like(data, dtype=np.uint8)
                
                return data
            except Exception as e:
                print(f"  Warning: Failed to extract patch at ({x:.1f}, {y:.1f}): {e}")
                return None
    
    def count_points_per_image(self, points_xy):
        """Count how many points fall in each image."""
        counts = {Path(img["path"]).name: 0 for img in self.images}
        orphans = 0
        for x, y in points_xy:
            img_path = self.find_image_for_point(x, y)
            if img_path:
                counts[Path(img_path).name] += 1
            else:
                orphans += 1
        return counts, orphans


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_ollama_running(base_url=OLLAMA_BASE_URL):
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models
        return False, []
    except requests.ConnectionError:
        return False, []


def check_model_available(model_name, base_url=OLLAMA_BASE_URL):
    running, models = check_ollama_running(base_url)
    if not running:
        return False, "Ollama is not running. Start it with: ollama serve"
    model_base = model_name.split(":")[0]
    for m in models:
        if m.startswith(model_base):
            return True, m
    return False, f"Model '{model_name}' not found. Pull it with: ollama pull {model_name}"


def image_to_base64(img_array, format="PNG"):
    if isinstance(img_array, np.ndarray):
        if img_array.ndim == 3 and img_array.shape[0] in [3, 4]:
            img_array = np.transpose(img_array, (1, 2, 0))
        if img_array.dtype in [np.float32, np.float64]:
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
        elif img_array.max() > 255:
            p2, p98 = np.percentile(img_array, (2, 98))
            img_array = np.clip((img_array - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)
        if img_array.ndim == 3 and img_array.shape[2] > 3:
            img_array = img_array[:, :, :3]
        img = Image.fromarray(img_array)
    elif isinstance(img_array, Image.Image):
        img = img_array
    else:
        raise ValueError(f"Unsupported image type: {type(img_array)}")
    buffer = io.BytesIO()
    img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def parse_vlm_response(response_text):
    response_text = response_text.strip()
    if "```json" in response_text:
        response_text = response_text.split("```json")[1].split("```")[0].strip()
    elif "```" in response_text:
        parts = response_text.split("```")
        if len(parts) >= 3:
            response_text = parts[1].strip()
    try:
        start = response_text.index("{")
        end = response_text.rindex("}") + 1
        result = json.loads(response_text[start:end])
        classification = result.get("classification", "ERROR").upper().strip()
        if "NOT" in classification:
            classification = "NOT_TUMULUS"
        elif "POSSIBLE" in classification:
            classification = "POSSIBLE_TUMULUS"
        elif "TUMULUS" in classification:
            classification = "TUMULUS"
        else:
            classification = "ERROR"
        return {
            "classification": classification,
            "confidence": float(result.get("confidence", 0)),
            "reasoning": str(result.get("reasoning", ""))[:500],
            "likely_feature": str(result.get("likely_feature", "unknown"))
        }
    except (json.JSONDecodeError, ValueError):
        response_upper = response_text.upper()
        if "NOT_TUMULUS" in response_upper or "NOT A TUMULUS" in response_upper:
            classification = "NOT_TUMULUS"
        elif "POSSIBLE" in response_upper:
            classification = "POSSIBLE_TUMULUS"
        elif "TUMULUS" in response_upper:
            classification = "TUMULUS"
        else:
            classification = "ERROR"
        return {"classification": classification, "confidence": 0.5,
                "reasoning": response_text[:500], "likely_feature": "unknown"}


# ============================================================
# EXTRACT-ONLY MODE
# ============================================================

def extract_patches_only(detections_path, image_paths, labels_path, output_dir,
                         labels_test_path=None, backgrounds_path=None,
                         patch_size=256, n_ref_positive=8, n_ref_negative=8):
    
    print("=" * 60)
    print("PATCH EXTRACTION v3 (multi-image support)")
    print("=" * 60)
    
    # ---- Initialize image manager ----
    print("\n[1/6] Loading images...")
    mgr = ImageManager(image_paths)
    
    # ---- Load detections ----
    print("\n[2/6] Loading CNN detections...")
    det_gdf = gpd.read_file(detections_path)
    if det_gdf.crs != mgr.crs:
        det_gdf = det_gdf.to_crs(mgr.crs)
    print(f"  {len(det_gdf)} detections loaded")
    
    # Show distribution
    det_points = [(r.geometry.x, r.geometry.y) for _, r in det_gdf.iterrows()]
    counts, orphans = mgr.count_points_per_image(det_points)
    for name, cnt in counts.items():
        print(f"    {name}: {cnt} detections")
    if orphans:
        print(f"    Outside all images: {orphans} (will be skipped)")
    
    # ---- Load labels ----
    print("\n[3/6] Loading tumulus labels...")
    labels_gdf = gpd.read_file(labels_path)
    if labels_test_path and os.path.exists(labels_test_path):
        test_gdf = gpd.read_file(labels_test_path)
        labels_gdf = pd.concat([labels_gdf, test_gdf], ignore_index=True)
        labels_gdf = gpd.GeoDataFrame(labels_gdf, crs=test_gdf.crs)
    if labels_gdf.crs != mgr.crs:
        labels_gdf = labels_gdf.to_crs(mgr.crs)
    print(f"  {len(labels_gdf)} labels loaded")
    
    # Show which labels fall in which image
    label_points = [(r.geometry.x, r.geometry.y) for _, r in labels_gdf.iterrows()]
    counts, orphans = mgr.count_points_per_image(label_points)
    for name, cnt in counts.items():
        print(f"    {name}: {cnt} labels")
    if orphans:
        print(f"    Outside all images: {orphans} (will be skipped!)")
    
    # ---- Create output dirs ----
    candidates_dir = os.path.join(output_dir, "candidates")
    ref_pos_dir = os.path.join(output_dir, "references", "positive")
    ref_neg_dir = os.path.join(output_dir, "references", "negative")
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(ref_pos_dir, exist_ok=True)
    os.makedirs(ref_neg_dir, exist_ok=True)
    
    # ---- Extract candidate patches ----
    print(f"\n[4/6] Extracting candidate patches (patch_size={patch_size}px)...")
    
    metadata_rows = []
    skipped = 0
    for idx, row in tqdm(det_gdf.iterrows(), total=len(det_gdf), desc="Candidates"):
        geom = row.geometry
        patch = mgr.extract_patch(geom.x, geom.y, patch_size)
        
        if patch is not None:
            fname = f"det_{idx:04d}.png"
            Image.fromarray(patch).save(os.path.join(candidates_dir, fname))
            meta = {"filename": fname, "x": geom.x, "y": geom.y, "det_index": idx}
            for col in det_gdf.columns:
                if col != "geometry":
                    meta[col] = row[col]
            metadata_rows.append(meta)
        else:
            skipped += 1
    
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    with open(os.path.join(output_dir, "crs.txt"), "w") as f:
        f.write(str(det_gdf.crs))
    
    print(f"  Saved {len(metadata_rows)} patches (skipped {skipped} out-of-bounds)")
    
    # ---- Extract POSITIVE reference patches ----
    print(f"\n[5/6] Extracting positive reference patches...")
    
    # Only sample from labels that actually fall within an image
    valid_indices = [i for i, (x, y) in enumerate(label_points)
                     if mgr.find_image_for_point(x, y) is not None]
    print(f"  {len(valid_indices)} of {len(labels_gdf)} labels within image bounds")
    
    random.seed(42)
    if len(valid_indices) > n_ref_positive:
        sample_indices = random.sample(valid_indices, n_ref_positive)
    else:
        sample_indices = valid_indices
    
    n_pos_saved = 0
    for i, idx in enumerate(sample_indices):
        row = labels_gdf.iloc[idx]
        patch = mgr.extract_patch(row.geometry.x, row.geometry.y, patch_size)
        if patch is not None:
            Image.fromarray(patch).save(os.path.join(ref_pos_dir, f"ref_pos_{i:02d}.png"))
            n_pos_saved += 1
    
    print(f"  Saved {n_pos_saved} positive reference patches")
    
    # ---- Extract NEGATIVE reference patches ----
    print(f"\n[6/6] Extracting negative reference patches...")
    
    negatives = []
    if backgrounds_path and os.path.exists(backgrounds_path):
        bg_gdf = gpd.read_file(backgrounds_path)
        if bg_gdf.crs != mgr.crs:
            bg_gdf = bg_gdf.to_crs(mgr.crs)
        
        # Filter to points within image bounds
        valid_bg = [i for i in range(len(bg_gdf))
                    if mgr.find_image_for_point(bg_gdf.iloc[i].geometry.x,
                                                 bg_gdf.iloc[i].geometry.y) is not None]
        print(f"  {len(valid_bg)} of {len(bg_gdf)} backgrounds within image bounds")
        
        random.seed(42)
        sample_bg = random.sample(valid_bg, min(n_ref_negative, len(valid_bg)))
        
        for idx in sample_bg:
            row = bg_gdf.iloc[idx]
            patch = mgr.extract_patch(row.geometry.x, row.geometry.y, patch_size)
            if patch is not None and patch.mean() > 10:
                negatives.append(patch)
    else:
        print("  No backgrounds.gpkg — using random locations")
        # Random sampling from each image
        for img_info in mgr.images:
            b = img_info["bounds"]
            margin = patch_size * 0.15 * 2
            random.seed(42)
            for _ in range(n_ref_negative * 20):
                if len(negatives) >= n_ref_negative:
                    break
                x = random.uniform(b.left + margin, b.right - margin)
                y = random.uniform(b.bottom + margin, b.top - margin)
                patch = mgr.extract_patch(x, y, patch_size)
                if patch is not None and patch.mean() > 10:
                    negatives.append(patch)
    
    for i, patch in enumerate(negatives):
        Image.fromarray(patch).save(os.path.join(ref_neg_dir, f"ref_neg_{i:02d}.png"))
    print(f"  Saved {len(negatives)} negative reference patches")
    
    # ---- Summary ----
    total_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(output_dir) for f in files
    ) / 1024 / 1024
    
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Images used:    {len(mgr.images)}")
    print(f"  Candidates:     {len(metadata_rows)} patches")
    print(f"  Ref positive:   {n_pos_saved} patches")
    print(f"  Ref negative:   {len(negatives)} patches")
    print(f"  Total size:     ~{total_size:.1f} MB")
    print(f"  Output:         {output_dir}")
    print(f"\n  Next: Upload to Drive → run Colab notebook")
    print(f"{'=' * 60}")


# ============================================================
# FULL VLM VERIFICATION
# ============================================================

def run_verification(
    detections_path, image_paths, labels_path, output_path,
    labels_test_path=None, backgrounds_path=None,
    model="llava", patch_size=256,
    n_ref_positive=8, n_ref_negative=8,
    confidence_filter=None, max_detections=None,
    save_patches=False, patches_dir=None, base_url=OLLAMA_BASE_URL,
):
    print("=" * 60)
    print("TUMULUS VLM VERIFICATION v3 (multi-image)")
    print("=" * 60)
    
    # Check Ollama
    print("\n[1/7] Checking Ollama...")
    available, model_info = check_model_available(model, base_url)
    if not available:
        print(f"  ERROR: {model_info}")
        sys.exit(1)
    print(f"  ✓ {model_info}")
    
    # Load images
    print("\n[2/7] Loading images...")
    mgr = ImageManager(image_paths)
    
    # Load detections
    print("\n[3/7] Loading detections...")
    gdf = gpd.read_file(detections_path)
    if gdf.crs != mgr.crs:
        gdf = gdf.to_crs(mgr.crs)
    n_total = len(gdf)
    print(f"  {n_total} detections")
    
    conf_col = None
    for col in ["confidence", "conf", "score", "prob"]:
        if col in gdf.columns:
            conf_col = col
            break
    
    if confidence_filter and conf_col:
        high_conf = gdf[gdf[conf_col] >= confidence_filter].copy()
        to_verify = gdf[gdf[conf_col] < confidence_filter].copy()
    else:
        high_conf = gpd.GeoDataFrame()
        to_verify = gdf.copy()
    
    if max_detections:
        to_verify = to_verify.head(max_detections)
    
    # Extract references
    print("\n[4/7] Extracting references...")
    labels_gdf = gpd.read_file(labels_path)
    if labels_test_path and os.path.exists(labels_test_path):
        test_gdf = gpd.read_file(labels_test_path)
        labels_gdf = pd.concat([labels_gdf, test_gdf], ignore_index=True)
        labels_gdf = gpd.GeoDataFrame(labels_gdf, crs=test_gdf.crs)
    if labels_gdf.crs != mgr.crs:
        labels_gdf = labels_gdf.to_crs(mgr.crs)
    
    valid_indices = [i for i in range(len(labels_gdf))
                     if mgr.find_image_for_point(labels_gdf.iloc[i].geometry.x,
                                                  labels_gdf.iloc[i].geometry.y) is not None]
    random.seed(42)
    sample_idx = random.sample(valid_indices, min(n_ref_positive, len(valid_indices)))
    
    ref_positive = []
    for idx in sample_idx:
        row = labels_gdf.iloc[idx]
        patch = mgr.extract_patch(row.geometry.x, row.geometry.y, patch_size)
        if patch is not None:
            ref_positive.append({"b64": image_to_base64(patch)})
    
    ref_negative = []
    if backgrounds_path and os.path.exists(backgrounds_path):
        bg_gdf = gpd.read_file(backgrounds_path)
        if bg_gdf.crs != mgr.crs:
            bg_gdf = bg_gdf.to_crs(mgr.crs)
        valid_bg = [i for i in range(len(bg_gdf))
                    if mgr.find_image_for_point(bg_gdf.iloc[i].geometry.x,
                                                 bg_gdf.iloc[i].geometry.y) is not None]
        random.seed(42)
        for idx in random.sample(valid_bg, min(n_ref_negative, len(valid_bg))):
            patch = mgr.extract_patch(bg_gdf.iloc[idx].geometry.x,
                                       bg_gdf.iloc[idx].geometry.y, patch_size)
            if patch is not None:
                ref_negative.append({"b64": image_to_base64(patch)})
    
    print(f"  {len(ref_positive)} positive, {len(ref_negative)} negative")
    
    # Build few-shot
    print("\n[5/7] Building few-shot prompt...")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i, ref in enumerate(ref_positive):
        messages.append({"role": "user",
                         "content": f"TRAINING EXAMPLE {i+1}: CONFIRMED TUMULUS.",
                         "images": [ref["b64"]]})
        messages.append({"role": "assistant",
                         "content": '{"classification": "TUMULUS", "confidence": 1.0, "reasoning": "Confirmed tumulus.", "likely_feature": "tumulus"}'})
    for i, ref in enumerate(ref_negative):
        messages.append({"role": "user",
                         "content": f"TRAINING EXAMPLE: NOT a tumulus. Background terrain.",
                         "images": [ref["b64"]]})
        messages.append({"role": "assistant",
                         "content": '{"classification": "NOT_TUMULUS", "confidence": 1.0, "reasoning": "Background terrain.", "likely_feature": "other"}'})
    
    if save_patches:
        if patches_dir is None:
            patches_dir = os.path.join(os.path.dirname(output_path), "vlm_patches")
        for s in ["tumulus", "possible_tumulus", "not_tumulus", "error"]:
            os.makedirs(os.path.join(patches_dir, s), exist_ok=True)
    
    # Run VLM
    print(f"\n[6/7] VLM verification on {len(to_verify)} detections...")
    results = []
    start_time = time.time()
    
    for idx, row in tqdm(to_verify.iterrows(), total=len(to_verify), desc="Verifying"):
        geom = row.geometry
        patch = mgr.extract_patch(geom.x, geom.y, patch_size)
        
        if patch is None:
            results.append({"vlm_class": "ERROR", "vlm_confidence": 0.0,
                          "vlm_reasoning": "Out of bounds", "vlm_feature": "unknown"})
            continue
        
        msgs = messages.copy()
        msgs.append({"role": "user", "content": CLASSIFICATION_PROMPT,
                      "images": [image_to_base64(patch)]})
        
        try:
            resp = requests.post(f"{base_url}/api/chat",
                                 json={"model": model, "messages": msgs, "stream": False,
                                       "options": {"temperature": 0.1, "num_predict": 300}},
                                 timeout=120)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
        except Exception as e:
            raw = f'{{"classification": "ERROR", "confidence": 0, "reasoning": "{e}"}}'
        
        parsed = parse_vlm_response(raw)
        results.append({"vlm_class": parsed["classification"],
                        "vlm_confidence": parsed["confidence"],
                        "vlm_reasoning": parsed["reasoning"],
                        "vlm_feature": parsed["likely_feature"]})
        
        if save_patches:
            cd = parsed["classification"].lower()
            if cd not in ["tumulus", "possible_tumulus", "not_tumulus"]:
                cd = "error"
            Image.fromarray(patch).save(os.path.join(patches_dir, cd, f"det_{idx:04d}.png"))
    
    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.0f}s")
    
    # Export
    print(f"\n[7/7] Exporting...")
    for key in ["vlm_class", "vlm_confidence", "vlm_reasoning", "vlm_feature"]:
        to_verify[key] = [r[key] for r in results]
    
    if not high_conf.empty:
        high_conf["vlm_class"] = "AUTO_ACCEPTED"
        high_conf["vlm_confidence"] = 1.0
        high_conf["vlm_reasoning"] = f"CNN conf >= {confidence_filter}"
        high_conf["vlm_feature"] = "tumulus"
        verified = gpd.GeoDataFrame(pd.concat([high_conf, to_verify], ignore_index=True), crs=gdf.crs)
    else:
        verified = to_verify
    
    verified.to_file(output_path, driver="GPKG")
    
    output_base, output_dir = Path(output_path).stem, Path(output_path).parent
    for subset, suffix in [
        (verified[verified["vlm_class"].isin(["TUMULUS", "AUTO_ACCEPTED"])], "_tumuli_only"),
        (verified[verified["vlm_class"].isin(["TUMULUS", "POSSIBLE_TUMULUS", "AUTO_ACCEPTED"])], "_tumuli_and_possible"),
    ]:
        if not subset.empty:
            subset.to_file(output_dir / f"{output_base}{suffix}.gpkg", driver="GPKG")
    
    counts = to_verify["vlm_class"].value_counts()
    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {n_total} detections")
    for cls in ["TUMULUS", "POSSIBLE_TUMULUS", "NOT_TUMULUS", "ERROR"]:
        print(f"  {cls:<20s}: {counts.get(cls, 0):4d}")
    print(f"{'=' * 60}")
    return verified


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Tumulus Verification v3")
    
    parser.add_argument("--detections", required=True)
    parser.add_argument("--image", required=True, nargs="+",
                        help="One or more Pleiades GeoTIFF paths")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--labels_test", default=None)
    parser.add_argument("--backgrounds", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="llava")
    parser.add_argument("--patch_size", type=int, default=256)
    parser.add_argument("--n_ref_positive", type=int, default=8)
    parser.add_argument("--n_ref_negative", type=int, default=8)
    parser.add_argument("--confidence_filter", type=float, default=None)
    parser.add_argument("--max_detections", type=int, default=None)
    parser.add_argument("--save_patches", action="store_true")
    parser.add_argument("--patches_dir", default=None)
    parser.add_argument("--ollama_url", default=OLLAMA_BASE_URL)
    parser.add_argument("--extract_only", action="store_true")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.detections):
        print(f"Error: Detections not found: {args.detections}"); sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"Error: Labels not found: {args.labels}"); sys.exit(1)
    
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    if args.extract_only:
        extract_patches_only(
            args.detections, args.image, args.labels, args.output,
            args.labels_test, args.backgrounds,
            args.patch_size, args.n_ref_positive, args.n_ref_negative)
    else:
        run_verification(
            args.detections, args.image, args.labels, args.output,
            args.labels_test, args.backgrounds,
            args.model, args.patch_size,
            args.n_ref_positive, args.n_ref_negative,
            args.confidence_filter, args.max_detections,
            args.save_patches, args.patches_dir, args.ollama_url)


if __name__ == "__main__":
    main()