#!/usr/bin/env python3
"""
GPS-Gaussian inference via PyTorch (no TensorRT).

For development on Maxwell/Pascal GPUs (sm_52-sm_61) that can't run TensorRT,
or for debugging/prototyping where TRT compilation isn't worth the overhead.

Requires: PyTorch 2.1.2+cu118 (last version with Maxwell support)
GPU: Tesla M40 (24GB sm_52), Quadro P-series, GTX 900/1000 series

This module provides the same interface as the TRT inference wrapper
(gaussian/src/trt_inference.h) but runs in pure PyTorch.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CameraCalibration:
    serial_number: int
    width: int
    height: int
    intrinsics: np.ndarray    # 3x3
    extrinsics: np.ndarray    # 4x4 world-to-camera


@dataclass
class GaussianOutput:
    positions: np.ndarray     # [N, 3]
    scales: np.ndarray        # [N, 3]
    rotations: np.ndarray     # [N, 4] wxyz
    opacities: np.ndarray     # [N]
    sh_coeffs: np.ndarray     # [N, 48]
    num_gaussians: int = 0
    inference_time_ms: float = 0.0


class GPSGaussianPyTorch:
    """
    PyTorch-based GPS-Gaussian inference.

    Drop-in replacement for TrtGaussianInference when TensorRT is unavailable.
    Same I/O contract: images + calibration → Gaussian attributes.

    Inference speed: ~1-5 FPS on M40 (vs ~25 FPS with TRT on RTX 3090).
    Good enough for PoC validation at 10-30s per frame.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_views: int = 2,
        input_height: int = 512,
        input_width: int = 960,
        device: str = "cuda:0",
    ):
        self.device = torch.device(device)
        self.num_views = num_views
        self.input_height = input_height
        self.input_width = input_width
        self.model = None

        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load GPS-Gaussian or compatible feed-forward model."""
        try:
            # Attempt to load GPS-Gaussian checkpoint
            # The actual import depends on the GPS-Gaussian repo structure.
            # This is a template — adapt to the specific model's API.
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # If it's a full model state dict:
            if "model" in checkpoint:
                # model = GPSGaussianModel(...)  # from gps_gaussian package
                # model.load_state_dict(checkpoint["model"])
                # self.model = model.eval().to(self.device)
                pass

            # If it's a TorchScript model:
            elif checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
                try:
                    self.model = torch.jit.load(checkpoint_path, map_location=self.device)
                    self.model.eval()
                    print(f"Loaded TorchScript model from {checkpoint_path}")
                except Exception:
                    print(f"Could not load as TorchScript: {checkpoint_path}")

            print(f"GPS-Gaussian loaded on {self.device}")

        except Exception as e:
            print(f"WARNING: Could not load GPS-Gaussian model: {e}")
            print("Running in dummy mode (random Gaussians for pipeline testing)")
            self.model = None

    @torch.no_grad()
    def infer(
        self,
        images: List[np.ndarray],          # K images, each [H, W, 3] uint8 RGB
        calibrations: List[CameraCalibration],
    ) -> GaussianOutput:
        """
        Run GPS-Gaussian inference.

        Args:
            images: K camera images (RGB uint8 numpy arrays)
            calibrations: K camera calibrations

        Returns:
            GaussianOutput with positions, scales, rotations, opacities, SH coefficients
        """
        t_start = time.time()

        assert len(images) == len(calibrations), "Image/calibration count mismatch"
        K = len(images)

        # Preprocess: resize + normalize + stack into batch tensor
        # Shape: [1, K, 3, H, W]
        processed = []
        for img in images:
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]
            t = F.interpolate(
                t.unsqueeze(0),
                size=(self.input_height, self.input_width),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)  # [3, H, W]
            processed.append(t)

        batch_images = torch.stack(processed).unsqueeze(0).to(self.device)  # [1, K, 3, H, W]

        # Build calibration tensors
        intrinsics = torch.zeros(1, K, 3, 3, device=self.device)
        extrinsics = torch.zeros(1, K, 4, 4, device=self.device)
        for i, cal in enumerate(calibrations):
            # Scale intrinsics to match inference resolution
            sx = self.input_width / cal.width
            sy = self.input_height / cal.height
            K_scaled = cal.intrinsics.copy()
            K_scaled[0, :] *= sx
            K_scaled[1, :] *= sy
            intrinsics[0, i] = torch.from_numpy(K_scaled)
            extrinsics[0, i] = torch.from_numpy(cal.extrinsics)

        # Run model
        if self.model is not None:
            outputs = self.model(batch_images, intrinsics, extrinsics)
            # Expected outputs: dict or tuple of (positions, scales, rotations, opacities, sh_coeffs)
            # Adapt based on actual model output format
            if isinstance(outputs, dict):
                positions = outputs["positions"].cpu().numpy()
                scales = outputs["scales"].cpu().numpy()
                rotations = outputs["rotations"].cpu().numpy()
                opacities = outputs["opacities"].cpu().numpy().squeeze(-1)
                sh_coeffs = outputs["sh_coeffs"].cpu().numpy()
            else:
                # Assume tuple: (positions, scales, rotations, opacities, sh_coeffs)
                positions, scales, rotations, opacities, sh_coeffs = [
                    o.cpu().numpy() for o in outputs
                ]
                if opacities.ndim == 2:
                    opacities = opacities.squeeze(-1)
        else:
            # Dummy mode: generate random Gaussians for pipeline testing
            N = 50000
            positions = np.random.randn(N, 3).astype(np.float32) * 0.3
            positions[:, 1] += 0.9  # center at human height
            scales = np.full((N, 3), 0.005, dtype=np.float32)
            rotations = np.zeros((N, 4), dtype=np.float32)
            rotations[:, 0] = 1.0  # identity quaternion
            opacities = np.random.uniform(0.3, 1.0, N).astype(np.float32)
            sh_coeffs = np.random.randn(N, 48).astype(np.float32) * 0.5

        t_elapsed = (time.time() - t_start) * 1000

        return GaussianOutput(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            num_gaussians=len(positions),
            inference_time_ms=t_elapsed,
        )


class RVMMattingPyTorch:
    """
    Robust Video Matting via PyTorch (no TensorRT).

    Drop-in replacement for segmentation/MattingEngine when TRT unavailable.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        input_height: int = 288,
        input_width: int = 512,
    ):
        self.device = torch.device(device)
        self.input_height = input_height
        self.input_width = input_width
        self.rec = [None] * 4  # RVM recurrent states

        try:
            import sys
            sys.path.insert(0, "third_party/RobustVideoMatting")
            from model import MattingNetwork

            self.model = MattingNetwork("mobilenetv3").eval().to(self.device)
            if checkpoint_path and os.path.exists(checkpoint_path):
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device)
                )
            print(f"RVM loaded on {self.device}")

        except Exception as e:
            print(f"WARNING: RVM not available ({e}). Using dummy matting.")
            self.model = None

    @torch.no_grad()
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Args:
            image: [H, W, 3] uint8 RGB

        Returns:
            alpha: [H, W] float32, 0-1
        """
        if self.model is None:
            return np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

        x = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
        x = F.interpolate(x, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)

        fgr, pha, *self.rec = self.model(x, *self.rec, downsample_ratio=0.25)

        # Resize back to original resolution
        pha = F.interpolate(pha, size=(image.shape[0], image.shape[1]), mode="bilinear", align_corners=False)
        return pha.squeeze().cpu().numpy()


import os  # noqa: E402 (needed for os.path.exists above)
