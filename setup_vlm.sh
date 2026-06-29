#!/bin/bash
# ============================================================
# VLM Tumulus Verification - Setup Script
# ============================================================
# Run this script to install all dependencies and verify
# that everything works before running the verification.
#
# Usage: bash setup_vlm.sh
# ============================================================

echo "=============================================="
echo "VLM Tumulus Verification - Setup"
echo "=============================================="
echo ""

# ---- Step 1: Check Python ----
echo "[1/5] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "  ✓ $PYTHON_VERSION"
else
    echo "  ✗ Python 3 not found. Install from python.org or via brew."
    exit 1
fi

# ---- Step 2: Install Python packages ----
echo ""
echo "[2/5] Installing Python packages..."
pip3 install --quiet geopandas rasterio numpy Pillow requests tqdm
if [ $? -eq 0 ]; then
    echo "  ✓ Python packages installed"
else
    echo "  ✗ Failed to install packages. Try: pip3 install geopandas rasterio numpy Pillow requests tqdm"
    exit 1
fi

# ---- Step 3: Check/Install Ollama ----
echo ""
echo "[3/5] Checking Ollama..."
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "installed")
    echo "  ✓ Ollama is installed ($OLLAMA_VERSION)"
else
    echo "  ✗ Ollama not found."
    echo ""
    echo "  Install Ollama:"
    echo "    Option A (Homebrew): brew install ollama"
    echo "    Option B (Direct):   curl -fsSL https://ollama.com/install.sh | sh"
    echo "    Option C (Manual):   Download from https://ollama.com/download"
    echo ""
    read -p "  Install via Homebrew now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        brew install ollama
    else
        echo "  Please install Ollama manually and re-run this script."
        exit 1
    fi
fi

# ---- Step 4: Start Ollama if not running ----
echo ""
echo "[4/5] Checking Ollama server..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✓ Ollama server is running"
else
    echo "  Starting Ollama server..."
    ollama serve &
    sleep 3
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "  ✓ Ollama server started"
    else
        echo "  ✗ Could not start Ollama. Try manually: ollama serve"
        exit 1
    fi
fi

# ---- Step 5: Pull vision model ----
echo ""
echo "[5/5] Checking vision models..."
echo ""
echo "  Available vision models (recommended for M1 Mac):"
echo "    1. llava (7B)        - ~4.7GB  - Good balance of speed/quality"
echo "    2. llava:13b          - ~8.0GB  - Better quality, slower"
echo "    3. llama3.2-vision    - ~7.9GB  - Latest Meta model, very capable"
echo "    4. llava:34b          - ~20GB   - Best quality but needs 32GB+ RAM"
echo ""
echo "  For M1 with 16GB RAM: llava (7B) or llama3.2-vision recommended"
echo "  For M1 with 8GB RAM:  llava (7B) only"
echo ""

# Check which models are already available
AVAILABLE=$(curl -s http://localhost:11434/api/tags | python3 -c "import sys,json; [print(f'    ✓ {m[\"name\"]}') for m in json.load(sys.stdin).get('models',[])]" 2>/dev/null)

if [ -n "$AVAILABLE" ]; then
    echo "  Currently installed models:"
    echo "$AVAILABLE"
    echo ""
fi

read -p "  Which model to use? (1=llava, 2=llava:13b, 3=llama3.2-vision, 4=llava:34b, or type name): " MODEL_CHOICE

case $MODEL_CHOICE in
    1) MODEL="llava" ;;
    2) MODEL="llava:13b" ;;
    3) MODEL="llama3.2-vision" ;;
    4) MODEL="llava:34b" ;;
    *) MODEL="$MODEL_CHOICE" ;;
esac

echo "  Pulling $MODEL (this may take a few minutes on first run)..."
ollama pull $MODEL

if [ $? -eq 0 ]; then
    echo "  ✓ Model $MODEL ready"
else
    echo "  ✗ Failed to pull model $MODEL"
    exit 1
fi

# ---- Quick test ----
echo ""
echo "=============================================="
echo "Setup complete! Quick verification test..."
echo "=============================================="
echo ""

python3 -c "
import requests, json
resp = requests.get('http://localhost:11434/api/tags')
models = [m['name'] for m in resp.json().get('models', [])]
vision_models = [m for m in models if any(v in m for v in ['llava', 'vision', 'gemma3'])]
print(f'  Ollama running: ✓')
print(f'  Vision models: {vision_models}')
print()
print('  Ready to run verification!')
print()
print(f'  Example command:')
print(f'  python3 vlm_verify_tumuli.py \\\\')
print(f'      --detections ./output/detections/tumulus_detections_pleiades1.gpkg \\\\')
print(f'      --image /path/to/pleiades1.TIF \\\\')
print(f'      --train_dir ./dataset_v3/train \\\\')
print(f'      --val_dir ./dataset_v3/val \\\\')
print(f'      --output ./output/detections/tumulus_verified_pleiades1.gpkg \\\\')
print(f'      --model $MODEL \\\\')
print(f'      --save_patches \\\\')
print(f'      --max_detections 10')
print()
print('  TIP: Start with --max_detections 10 to test, then remove for full run.')
"

echo ""
echo "=============================================="