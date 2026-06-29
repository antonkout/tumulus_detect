# Tumulus Detection — Somalia (Pleiades Neo)

Detecting ancient stone tumuli (burial cairns) from very-high-resolution
satellite imagery (Pleiades Neo HD, 15 cm/px) over arid landscapes in the
Horn of Africa.

## Approach

A **two-stage pipeline**: a high-recall CNN proposer followed by a
vision-language-model verifier. Ground-truth labels are scarce (~32 points),
so the design casts a wide net with YOLO and culls false positives with a VLM.

```
Pleiades GeoTIFF
      │
      ▼
┌─────────────────────────┐
│ Stage 1 — YOLOv8         │  sliding-window detection
│ (tumulus_detect.py)      │  • multi-band combos: RGB + CIR + NDVI
│                          │  • 5-way test-time augmentation
│                          │  • nodata-tile skipping
│                          │  • geographic-distance NMS + evaluation
└───────────┬─────────────┘
            │  candidate points (GeoPackage) — high recall, low precision
            ▼
┌─────────────────────────┐
│ Stage 2 — VLM verify     │  few-shot LLaVA via Ollama
│ (vlm_verify_tumuli.py)   │  • positive/negative reference patches in-context
│                          │  • classifies each candidate:
│                          │    TUMULUS / POSSIBLE_TUMULUS / NOT_TUMULUS
└───────────┬─────────────┘
            │
            ▼
   Verified tumuli (GeoPackage) → QGIS review
```

### Stage 1 — YOLOv8 detection
- Trained on 22 positive points + hard negatives (see `tumulus_yolo_v3.ipynb`).
- At inference, each tile is run through three band combinations (RGB, CIR,
  NDVI) and detections are merged before non-max suppression.
- Pleiades Neo bands are **RGBN** (Red=0, Green=1, Blue=2, NIR=3); training and
  inference must use this order consistently.
- Output: detections as GeoPackage, plus precision/recall/F1 sweeps against the
  held-out test set.

### Stage 2 — VLM verification
- Each YOLO candidate patch is shown to a local vision LLM (LLaVA via Ollama)
  with a few-shot prompt of confirmed tumuli and background examples.
- The VLM does the fine discrimination YOLO can't (vegetation, rock outcrops,
  shadows), turning a noisy candidate list into a clean one.
- Runs fully offline — no API keys required.

## Repository layout

```
tumulus_detect.py        Stage 1 — YOLO sliding-window inference + evaluation
vlm_verify_tumuli.py     Stage 2 — VLM verification pipeline
create_patches.ipynb     Patch extraction helper
tumulus_yolo_v3.ipynb    Training notebook (Colab)
setup_vlm.sh             Ollama + vision-model setup
environment/             Conda + Poetry environment specs
CHANGES.md               Version history
```

> **Note:** satellite imagery, model weights, and tumulus coordinates are not
> included in this repo — the imagery is licensed (Airbus) and site locations
> are withheld to reduce looting risk.

## Setup

Conda:
```bash
conda env create -f environment/environment.yml
conda activate tumulus
```

or Poetry:
```bash
cd environment && poetry install
```

For Stage 2, install Ollama and a vision model:
```bash
bash setup_vlm.sh
```

## Usage

Stage 1 — run detection (configure paths at the top of the file):
```bash
python tumulus_detect.py
```

Stage 2 — verify candidates:
```bash
python vlm_verify_tumuli.py \
    --detections ./output/detections/tumulus_detections.gpkg \
    --image ./data/pleiades1.TIF ./data/pleiades3.TIF \
    --labels ./data/tombs_training.gpkg \
    --backgrounds ./output/detections/backgrounds.gpkg \
    --output ./output/detections/verified.gpkg \
    --model llava
```
