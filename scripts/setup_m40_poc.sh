#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Heimdall M40 PoC Setup
#
# Sets up a development environment on Tesla M40 (sm_52, 24GB VRAM)
# using CUDA 11.8 + PyTorch 2.1 (last versions with Maxwell support).
#
# This bypasses TensorRT entirely — inference runs in PyTorch directly.
# Slower than production (10-30s/frame vs real-time) but functionally
# identical for proof-of-concept validation.
# ============================================================================

echo "=== Heimdall M40 PoC Setup ==="
echo "GPU: Tesla M40 (sm_52, 24GB)"
echo "Stack: CUDA 11.8 + PyTorch 2.1.2 + cu118"
echo ""

# Check GPU
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
echo "Detected GPU: $GPU_NAME ($GPU_MEM)"

# ---- Python Environment ----
echo ""
echo "=== Creating Python environment ==="

python3 -m venv .venv-m40
source .venv-m40/bin/activate

pip install --upgrade pip

# ---- PyTorch 2.1.2 with CUDA 11.8 (last Maxwell support) ----
echo ""
echo "=== Installing PyTorch 2.1.2+cu118 ==="
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA works on this GPU
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
    # Quick test
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'Matrix multiply test: OK ({y.shape})')
else:
    print('ERROR: CUDA not available')
    exit(1)
"

# ---- Core dependencies ----
echo ""
echo "=== Installing dependencies ==="
pip install \
    opencv-python-headless \
    numpy \
    Pillow \
    trimesh \
    jsonschema \
    pyyaml \
    protobuf \
    grpcio \
    grpcio-tools \
    tensorboard \
    lpips \
    onnx \
    onnxruntime  # CPU ONNX runtime (no TRT dependency)

# ---- GPS-Gaussian ----
echo ""
echo "=== Installing GPS-Gaussian ==="
if [ ! -d "third_party/GPS-Gaussian" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/aipixel/GPS-Gaussian.git
    cd GPS-Gaussian
    pip install -e . 2>/dev/null || echo "GPS-Gaussian: manual integration needed (no setup.py)"
    cd ../..
fi

# ---- Robust Video Matting (RVM) ----
echo ""
echo "=== Installing RVM ==="
if [ ! -d "third_party/RobustVideoMatting" ]; then
    mkdir -p third_party
    cd third_party
    git clone https://github.com/PeterL1n/RobustVideoMatting.git
    cd ../..
fi

# ---- gsplat (CUDA Gaussian rasterizer) ----
echo ""
echo "=== Installing gsplat ==="
# gsplat needs to compile CUDA kernels — force sm_52
TORCH_CUDA_ARCH_LIST="5.2" pip install gsplat==0.1.11 2>/dev/null || \
    echo "gsplat: may need manual build with CUDA 11.8 for sm_52"

# ---- Summary ----
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate:  source .venv-m40/bin/activate"
echo ""
echo "PoC differences from production:"
echo "  - Inference via PyTorch (not TensorRT) — slower but identical output"
echo "  - CUDA 11.8 (not 12.4) — all kernels compile for sm_52"
echo "  - No NVENC (M40 lacks encoder) — skip H.265 encoding tests"
echo "  - ~10-30s per frame (vs real-time on H100)"
echo ""
echo "To run GPS-Gaussian PoC:"
echo "  python scripts/poc_gaussian_preview.py --gpu 0"
