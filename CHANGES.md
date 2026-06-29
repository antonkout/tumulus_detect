# Tumulus Detection — v2 Changes

## Label split strategy
- **Training (22 points)**  → `train/images` — YOLO trains on these
- **Monitoring (10 points)** → `val/images`  — YOLO monitors loss each epoch (never trained on)
  - Test (6 clean points) + Validation (4 extreme/uncertain points) merged into `monitor_gdf`
- **Test (6 points)** → used ONLY in Cell 22 post-training evaluation for final recall metric

## tumulus_yolo_pipeline.ipynb — cells changed

| Cell | Change |
|------|--------|
| 4  | Added `VAL_LABELS_PATH`, `NODATA_THRESHOLD`, `USE_BAND_COMBOS`, `BAND_ORDER`; epochs 350, augment_factor 15 |
| 7  | Loads all 3 GeoPackages; merges test+val into `monitor_gdf` (10 points) |
| 9  | Visualises val and test label sets separately |
| 12 | Full replacement: `prepare_dataset_v2()` with nodata filtering, edge clamping, NIR band combos |
| 13 | Calls `prepare_dataset_v2(train_labels_gdf=train_gdf, val_labels_gdf=monitor_gdf)` |
| 17 | `degrees=180`, `scale=0.5`, `patience=60`, output → `train_v3` |
| 18 | Resume cell fixed (no more hardcoded absolute path) |
| 19 | Results dir fixed → `train_v3` |
| 21 | Model load fixed → `train_v3/weights/best.pt` |
| 22 | Evaluation uses `test_gdf` only (6 clean points); band read uses `BAND_ORDER` |

## tumulus_detect.py — changes
- Multi-band inference: RGB + CIR + NDVI per tile, detections merged before NMS
- NoData tile skipping (>15% black pixels → skip)
- Separate evaluation for test and validation label sets
- `BAND_ORDER` dict configurable at top of file
- `VAL_LABELS_PATH` support alongside `TEST_LABELS_PATH`

## tumulus_dataset_v2.py
Standalone module version of `prepare_dataset_v2()` for use outside the notebook.
