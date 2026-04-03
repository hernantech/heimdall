#!/usr/bin/env python3
"""
Export Robust Video Matting (RVM) or BackgroundMattingV2 (BMV2) to ONNX.

Usage:
    # Export RVM at 512x288 (default):
    python export_matting_model.py --model rvm --output rvm.onnx

    # Export BMV2 at 1024x576:
    python export_matting_model.py --model bmv2 --output bmv2.onnx \
        --width 1024 --height 576

    # Export and verify with onnxruntime:
    python export_matting_model.py --model rvm --output rvm.onnx --verify

    # Convert ONNX to TensorRT (requires trtexec on PATH):
    python export_matting_model.py --model rvm --output rvm.onnx --export-trt

Requirements:
    pip install torch torchvision onnx onnxruntime-gpu numpy

The exported ONNX model uses opset 17 with dynamic batch dimension.
Input/output naming conventions match what matting.cpp expects:
    RVM:   input="input" [B,3,H,W],  output alpha="pha" [B,1,H,W]
    BMV2:  input="src"   [B,6,H,W],  output alpha="pha" [B,1,H,W],
                                             foreground="fgr" [B,3,H,W]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ============================================================================
# Model wrappers
# ============================================================================

class RVMExportWrapper(nn.Module):
    """
    Wraps Robust Video Matting for ONNX export.

    RVM internally uses recurrent hidden states for temporal coherence.
    For ONNX export, we use the model in single-frame mode (no recurrence)
    since temporal smoothing is handled in our C++ postprocessing pipeline.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] RGB float32, range [0, 1]
        Returns:
            alpha: [B, 1, H, W] float32, range [0, 1]
        """
        # RVM forward returns (fgr, pha, *rec) in evaluation mode.
        # We only need the alpha (pha) channel.
        # Initialize recurrent states to None for single-frame mode.
        fgr, pha, *_rec = self.model(x, *([None] * 4), downsample_ratio=0.25)
        return pha


class BMV2ExportWrapper(nn.Module):
    """
    Wraps BackgroundMattingV2 for ONNX export.

    BMV2 takes a 6-channel input: [B, 6, H, W] where channels 0-2 are
    the foreground RGB and channels 3-5 are the background (clean plate) RGB.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            src: [B, 6, H, W] float32, range [0, 1]
                 channels 0-2 = foreground RGB
                 channels 3-5 = background RGB
        Returns:
            alpha: [B, 1, H, W] float32, range [0, 1]
            fgr:   [B, 3, H, W] float32, predicted foreground
        """
        foreground = src[:, :3, :, :]
        background = src[:, 3:, :, :]
        pha, fgr = self.model(foreground, background)[:2]
        return pha, fgr


# ============================================================================
# Model loading
# ============================================================================

def load_rvm_model(device: torch.device) -> nn.Module:
    """
    Load Robust Video Matting from torchub or local checkpoint.

    Tries torch.hub first; falls back to local checkpoint if available.
    """
    print("Loading Robust Video Matting (RVM)...")

    # Try loading from torch.hub (PeterL1n/RobustVideoMatting)
    try:
        model = torch.hub.load(
            "PeterL1n/RobustVideoMatting",
            "mobilenetv3",
            pretrained=True,
        )
        model = model.to(device).eval()
        print("  Loaded RVM MobileNetV3 from torch.hub")
        return RVMExportWrapper(model)
    except Exception as e:
        print(f"  torch.hub load failed: {e}")

    # Try loading from local checkpoint
    ckpt_paths = [
        Path("rvm_mobilenetv3.pth"),
        Path(__file__).parent.parent / "models" / "rvm_mobilenetv3.pth",
    ]
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            print(f"  Loading from local checkpoint: {ckpt_path}")
            model = torch.hub.load(
                "PeterL1n/RobustVideoMatting",
                "mobilenetv3",
                pretrained=False,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model = model.to(device).eval()
            return RVMExportWrapper(model)

    raise RuntimeError(
        "Could not load RVM model. Either:\n"
        "  1. Ensure network access for torch.hub download, or\n"
        "  2. Place rvm_mobilenetv3.pth in segmentation/models/"
    )


def load_bmv2_model(device: torch.device) -> nn.Module:
    """
    Load BackgroundMattingV2 from torch.hub or local checkpoint.
    """
    print("Loading BackgroundMattingV2 (BMV2)...")

    try:
        model = torch.hub.load(
            "PeterL1n/BackgroundMattingV2",
            "mobilenetv2",
            pretrained=True,
        )
        model = model.to(device).eval()
        print("  Loaded BMV2 MobileNetV2 from torch.hub")
        return BMV2ExportWrapper(model)
    except Exception as e:
        print(f"  torch.hub load failed: {e}")

    ckpt_paths = [
        Path("bmv2_mobilenetv2.pth"),
        Path(__file__).parent.parent / "models" / "bmv2_mobilenetv2.pth",
    ]
    for ckpt_path in ckpt_paths:
        if ckpt_path.exists():
            print(f"  Loading from local checkpoint: {ckpt_path}")
            model = torch.hub.load(
                "PeterL1n/BackgroundMattingV2",
                "mobilenetv2",
                pretrained=False,
            )
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model = model.to(device).eval()
            return BMV2ExportWrapper(model)

    raise RuntimeError(
        "Could not load BMV2 model. Either:\n"
        "  1. Ensure network access for torch.hub download, or\n"
        "  2. Place bmv2_mobilenetv2.pth in segmentation/models/"
    )


# ============================================================================
# ONNX export
# ============================================================================

def export_rvm_onnx(
    model: nn.Module,
    output_path: str,
    height: int,
    width: int,
    device: torch.device,
    opset: int = 17,
) -> None:
    """Export RVM wrapper to ONNX."""
    print(f"Exporting RVM to ONNX: {output_path}")
    print(f"  Resolution: {width}x{height}, opset: {opset}")

    dummy_input = torch.randn(1, 3, height, width, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["pha"],
        dynamic_axes={
            "input": {0: "batch"},
            "pha":   {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"  Exported: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def export_bmv2_onnx(
    model: nn.Module,
    output_path: str,
    height: int,
    width: int,
    device: torch.device,
    opset: int = 17,
) -> None:
    """Export BMV2 wrapper to ONNX."""
    print(f"Exporting BMV2 to ONNX: {output_path}")
    print(f"  Resolution: {width}x{height}, opset: {opset}")

    # 6-channel input: foreground RGB + background RGB
    dummy_input = torch.randn(1, 6, height, width, device=device)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=["src"],
        output_names=["pha", "fgr"],
        dynamic_axes={
            "src": {0: "batch"},
            "pha": {0: "batch"},
            "fgr": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"  Exported: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ============================================================================
# ONNX verification
# ============================================================================

def verify_onnx(onnx_path: str, model_type: str, height: int, width: int) -> None:
    """Verify the exported ONNX model with onnxruntime."""
    print(f"\nVerifying ONNX model: {onnx_path}")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("  Skipping verification: onnx or onnxruntime not installed")
        print("  Install with: pip install onnx onnxruntime-gpu")
        return

    # Validate ONNX graph structure
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    print("  ONNX graph validation: PASSED")

    # Print model info
    print(f"  IR version: {onnx_model.ir_version}")
    print(f"  Opset: {onnx_model.opset_import[0].version}")
    print("  Inputs:")
    for inp in onnx_model.graph.input:
        shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in inp.type.tensor_type.shape.dim
        ]
        print(f"    {inp.name}: {shape}")
    print("  Outputs:")
    for out in onnx_model.graph.output:
        shape = [
            d.dim_param if d.dim_param else d.dim_value
            for d in out.type.tensor_type.shape.dim
        ]
        print(f"    {out.name}: {shape}")

    # Run inference with onnxruntime
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [p for p in providers if p in available]
    print(f"  onnxruntime providers: {providers}")

    session = ort.InferenceSession(onnx_path, providers=providers)

    if model_type == "rvm":
        dummy = np.random.rand(1, 3, height, width).astype(np.float32)
        outputs = session.run(None, {"input": dummy})
        alpha = outputs[0]
        print(f"  Output alpha shape: {alpha.shape}")
        print(f"  Output alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
        assert alpha.shape == (1, 1, height, width), (
            f"Unexpected alpha shape: {alpha.shape}"
        )
    else:
        dummy = np.random.rand(1, 6, height, width).astype(np.float32)
        outputs = session.run(None, {"src": dummy})
        alpha = outputs[0]
        fgr = outputs[1]
        print(f"  Output alpha shape: {alpha.shape}")
        print(f"  Output fgr shape:   {fgr.shape}")
        print(f"  Output alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
        assert alpha.shape == (1, 1, height, width), (
            f"Unexpected alpha shape: {alpha.shape}"
        )
        assert fgr.shape == (1, 3, height, width), (
            f"Unexpected fgr shape: {fgr.shape}"
        )

    print("  onnxruntime inference: PASSED")


# ============================================================================
# TensorRT conversion
# ============================================================================

def convert_to_tensorrt(onnx_path: str, trt_path: str | None = None) -> str:
    """
    Convert ONNX model to TensorRT engine using trtexec.

    Args:
        onnx_path: Path to the .onnx model
        trt_path:  Output .trt path (default: same name with .trt extension)

    Returns:
        Path to the generated .trt engine file.
    """
    if trt_path is None:
        trt_path = onnx_path.replace(".onnx", ".trt")

    print(f"\nConverting to TensorRT: {onnx_path} -> {trt_path}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--fp16",                    # Enable FP16 for ~2x speedup
        "--workspace=4096",          # 4 GB workspace
        "--minShapes=input:1x3x288x512",   # Adjust if needed
        "--optShapes=input:1x3x288x512",
        "--maxShapes=input:1x3x576x1024",
        "--verbose",
    ]

    print(f"  Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,  # 10 minute timeout for engine build
        )
        print(f"  TensorRT engine saved: {trt_path}")
        print(f"  Size: {os.path.getsize(trt_path) / 1e6:.1f} MB")
        return trt_path
    except FileNotFoundError:
        print("  ERROR: trtexec not found on PATH")
        print("  Install TensorRT and ensure trtexec is available")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: trtexec failed with code {e.returncode}")
        print(f"  stderr: {e.stderr[-2000:]}")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("  ERROR: trtexec timed out after 10 minutes")
        sys.exit(1)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export matting model (RVM or BMV2) to ONNX for Heimdall pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model",
        choices=["rvm", "bmv2"],
        default="rvm",
        help="Model architecture to export (default: rvm)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output ONNX file path (e.g., rvm.onnx)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Model input width (default: 512)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=288,
        help="Model input height (default: 288)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the exported model with onnxruntime",
    )
    parser.add_argument(
        "--export-trt",
        action="store_true",
        help="Also convert to TensorRT engine (requires trtexec)",
    )
    parser.add_argument(
        "--trt-output",
        type=str,
        default=None,
        help="TensorRT engine output path (default: same as ONNX with .trt extension)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device (default: cuda)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    if device.type == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    # Load model
    if args.model == "rvm":
        model = load_rvm_model(device)
    else:
        model = load_bmv2_model(device)

    # Export to ONNX
    with torch.no_grad():
        if args.model == "rvm":
            export_rvm_onnx(
                model, args.output, args.height, args.width, device, args.opset
            )
        else:
            export_bmv2_onnx(
                model, args.output, args.height, args.width, device, args.opset
            )

    # Verify
    if args.verify:
        verify_onnx(args.output, args.model, args.height, args.width)

    # TensorRT conversion
    if args.export_trt:
        trt_path = args.trt_output or args.output.replace(".onnx", ".trt")
        convert_to_tensorrt(args.output, trt_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
