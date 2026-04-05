#!/usr/bin/env python3
"""
Heimdall Test Harness — Replay multi-camera captures through the Gaussian pipeline.

Reads recorded multi-camera video streams (one file per camera) alongside
calibration data, decodes frame-by-frame, and feeds them through a
pluggable inference backend (PyTorch/M40, TensorRT, or remote RunPod workers).

Usage:
    # Local PyTorch backend (M40/dev GPU)
    python harness/scripts/run_harness.py \
        --data /path/to/capture/ \
        --calibration /path/to/rig.json \
        --backend pytorch \
        --checkpoint /path/to/model.pth \
        --output /path/to/output/ \
        --gpu 0

    # Local TensorRT backend (RTX 3090+)
    python harness/scripts/run_harness.py \
        --data /path/to/capture/ \
        --calibration /path/to/rig.json \
        --backend tensorrt \
        --engine /path/to/model.trt \
        --output /path/to/output/

    # Remote RunPod workers
    python harness/scripts/run_harness.py \
        --data /path/to/capture/ \
        --calibration /path/to/rig.json \
        --backend runpod \
        --workers worker1.ts.net:50051,worker2.ts.net:50051 \
        --output /path/to/output/

Data directory layout:
    capture/
        cam_00/         (or cam_00.mp4 — both supported)
            frame_000000.jpg
            frame_000001.jpg
            ...
        cam_01/
            ...
        cam_19/
            ...

    -or-

    capture/
        cam_00.mp4      (one video file per camera)
        cam_01.mp4
        ...

Calibration: heimdall rig.schema.json format (OpenCV convention).
"""

import argparse
import glob
import json
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraCalibration:
    index: int
    serial_number: int
    width: int
    height: int
    intrinsics: np.ndarray    # 3x3
    extrinsics: np.ndarray    # 4x4 world-to-camera
    dist_coeffs: Optional[np.ndarray] = None


@dataclass
class CameraFrame:
    camera_index: int
    frame_id: int
    timestamp_ns: int
    image: np.ndarray         # [H, W, 3] uint8 BGR (OpenCV convention)


@dataclass
class SyncedFrameSet:
    frame_id: int
    timestamp_ns: int
    frames: Dict[int, CameraFrame]  # camera_index → frame


@dataclass
class GaussianResult:
    frame_id: int
    num_gaussians: int
    positions: np.ndarray     # [N, 3]
    scales: np.ndarray        # [N, 3]
    rotations: np.ndarray     # [N, 4]
    opacities: np.ndarray     # [N]
    sh_coeffs: np.ndarray     # [N, 48]
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Multi-camera stream reader
# ---------------------------------------------------------------------------

class MultiCameraReader:
    """
    Reads synchronized frames from multiple camera sources.
    Supports:
      - Directory of image sequences (cam_XX/frame_NNNNNN.jpg)
      - Video files per camera (cam_XX.mp4)
      - Mixed formats
    """

    def __init__(self, data_path: str, calibrations: List[CameraCalibration]):
        self.data_path = Path(data_path)
        self.calibrations = {c.index: c for c in calibrations}
        self.sources = {}  # camera_index → source (VideoCapture or frame list)
        self.frame_count = 0
        self._discover_sources()

    def _discover_sources(self):
        """Find camera sources in the data directory."""
        # Try video files first
        video_exts = [".mp4", ".avi", ".mkv", ".mov", ".h264", ".h265", ".hevc"]
        for ext in video_exts:
            for f in sorted(self.data_path.glob(f"cam_*{ext}")):
                idx = int(f.stem.split("_")[1])
                cap = cv2.VideoCapture(str(f))
                if cap.isOpened():
                    self.sources[idx] = {"type": "video", "cap": cap, "path": str(f)}
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.frame_count = max(self.frame_count, count)
                    print(f"  Camera {idx}: {f.name} ({count} frames)")

        # Try image sequence directories
        for d in sorted(self.data_path.glob("cam_*")):
            if d.is_dir():
                idx = int(d.name.split("_")[1])
                if idx in self.sources:
                    continue  # already found as video
                frames = sorted(glob.glob(str(d / "*.jpg")) +
                                glob.glob(str(d / "*.png")) +
                                glob.glob(str(d / "*.exr")) +
                                glob.glob(str(d / "*.tiff")))
                if frames:
                    self.sources[idx] = {"type": "images", "frames": frames}
                    self.frame_count = max(self.frame_count, len(frames))
                    print(f"  Camera {idx}: {d.name}/ ({len(frames)} frames)")

        if not self.sources:
            raise FileNotFoundError(f"No camera sources found in {self.data_path}")

        print(f"  Total: {len(self.sources)} cameras, {self.frame_count} frames")

    def read_frame(self, frame_id: int) -> SyncedFrameSet:
        """Read one synchronized frame set across all cameras."""
        frames = {}
        timestamp_ns = int(frame_id * (1e9 / 30))  # assume 30fps

        for cam_idx, source in self.sources.items():
            if source["type"] == "video":
                cap = source["cap"]
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, img = cap.read()
                if not ret:
                    continue
            elif source["type"] == "images":
                if frame_id >= len(source["frames"]):
                    continue
                img = cv2.imread(source["frames"][frame_id])
                if img is None:
                    continue

            frames[cam_idx] = CameraFrame(
                camera_index=cam_idx,
                frame_id=frame_id,
                timestamp_ns=timestamp_ns,
                image=img,
            )

        return SyncedFrameSet(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            frames=frames,
        )

    def __len__(self):
        return self.frame_count

    def close(self):
        for source in self.sources.values():
            if source["type"] == "video":
                source["cap"].release()


# ---------------------------------------------------------------------------
# Backend interface (pluggable)
# ---------------------------------------------------------------------------

class InferenceBackend(ABC):
    """Abstract backend — swap implementations without changing the harness."""

    @abstractmethod
    def initialize(self, calibrations: List[CameraCalibration], **kwargs):
        """Load models, allocate buffers."""
        pass

    @abstractmethod
    def process_frame(
        self,
        frame_set: SyncedFrameSet,
        calibrations: Dict[int, CameraCalibration],
    ) -> GaussianResult:
        """Process one synced frame set → Gaussian output."""
        pass

    @abstractmethod
    def shutdown(self):
        """Release resources."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class PyTorchBackend(InferenceBackend):
    """Local PyTorch inference (M40 / any GPU with CUDA 11.8+)."""

    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        self.checkpoint = checkpoint
        self.device = device
        self.gs_model = None
        self.matting_model = None

    @property
    def name(self):
        return f"pytorch ({self.device})"

    def initialize(self, calibrations, **kwargs):
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "inference" / "pytorch"))
        from gps_gaussian_pt import GPSGaussianPyTorch, RVMMattingPyTorch

        self.gs_model = GPSGaussianPyTorch(
            checkpoint_path=self.checkpoint,
            device=self.device,
            num_views=2,
        )
        matting_ckpt = kwargs.get("matting_checkpoint", "")
        self.matting_model = RVMMattingPyTorch(
            checkpoint_path=matting_ckpt,
            device=self.device,
        )
        print(f"PyTorch backend initialized on {self.device}")

    def process_frame(self, frame_set, calibrations):
        from gps_gaussian_pt import CameraCalibration as PTCalib

        t_start = time.time()

        # Sort cameras by index
        cam_indices = sorted(frame_set.frames.keys())
        if len(cam_indices) < 2:
            return GaussianResult(
                frame_id=frame_set.frame_id, num_gaussians=0,
                positions=np.empty((0, 3)), scales=np.empty((0, 3)),
                rotations=np.empty((0, 4)), opacities=np.empty(0),
                sh_coeffs=np.empty((0, 48)), processing_time_ms=0,
            )

        # Select stereo pairs (adjacent cameras)
        pairs = []
        for i in range(0, len(cam_indices) - 1, 2):
            pairs.append((cam_indices[i], cam_indices[i + 1]))

        all_positions = []
        all_scales = []
        all_rotations = []
        all_opacities = []
        all_sh = []

        for cam_a_idx, cam_b_idx in pairs:
            frame_a = frame_set.frames[cam_a_idx]
            frame_b = frame_set.frames[cam_b_idx]
            cal_a = calibrations[cam_a_idx]
            cal_b = calibrations[cam_b_idx]

            # Convert BGR→RGB
            img_a = cv2.cvtColor(frame_a.image, cv2.COLOR_BGR2RGB)
            img_b = cv2.cvtColor(frame_b.image, cv2.COLOR_BGR2RGB)

            # Matting
            mask_a = self.matting_model.process(img_a)
            mask_b = self.matting_model.process(img_b)

            # Apply masks (zero out background)
            img_a = (img_a * mask_a[:, :, None]).astype(np.uint8)
            img_b = (img_b * mask_b[:, :, None]).astype(np.uint8)

            # GPS-Gaussian inference on stereo pair
            pt_cals = [
                PTCalib(cal_a.serial_number, cal_a.width, cal_a.height,
                        cal_a.intrinsics, cal_a.extrinsics),
                PTCalib(cal_b.serial_number, cal_b.width, cal_b.height,
                        cal_b.intrinsics, cal_b.extrinsics),
            ]
            result = self.gs_model.infer([img_a, img_b], pt_cals)

            all_positions.append(result.positions)
            all_scales.append(result.scales)
            all_rotations.append(result.rotations)
            all_opacities.append(result.opacities)
            all_sh.append(result.sh_coeffs)

        # Concatenate results from all pairs
        if all_positions:
            positions = np.concatenate(all_positions)
            scales = np.concatenate(all_scales)
            rotations = np.concatenate(all_rotations)
            opacities = np.concatenate(all_opacities)
            sh_coeffs = np.concatenate(all_sh)
        else:
            positions = np.empty((0, 3), dtype=np.float32)
            scales = np.empty((0, 3), dtype=np.float32)
            rotations = np.empty((0, 4), dtype=np.float32)
            opacities = np.empty(0, dtype=np.float32)
            sh_coeffs = np.empty((0, 48), dtype=np.float32)

        elapsed = (time.time() - t_start) * 1000

        return GaussianResult(
            frame_id=frame_set.frame_id,
            num_gaussians=len(positions),
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            sh_coeffs=sh_coeffs,
            processing_time_ms=elapsed,
        )

    def shutdown(self):
        del self.gs_model
        del self.matting_model
        import torch
        torch.cuda.empty_cache()


class RunPodBackend(InferenceBackend):
    """Remote RunPod workers via gRPC over Tailscale."""

    def __init__(self, worker_addresses: List[str]):
        self.worker_addresses = worker_addresses
        self.channels = []

    @property
    def name(self):
        return f"runpod ({len(self.worker_addresses)} workers)"

    def initialize(self, calibrations, **kwargs):
        # In production: create gRPC channels to each worker
        # import grpc
        # from worker.proto import worker_pb2_grpc
        # for addr in self.worker_addresses:
        #     channel = grpc.insecure_channel(addr)
        #     stub = worker_pb2_grpc.GaussianWorkerStub(channel)
        #     self.channels.append((channel, stub))
        print(f"RunPod backend: {len(self.worker_addresses)} workers")
        for addr in self.worker_addresses:
            print(f"  {addr}")

    def process_frame(self, frame_set, calibrations):
        t_start = time.time()

        # In production:
        # 1. Encode frames as H.265 NAL units (or send raw JPEG)
        # 2. Split stereo pairs across workers
        # 3. Send ProcessFrameRequest to each worker via gRPC
        # 4. Collect WorkerFrameResponse from all workers
        # 5. Dequantize and concatenate

        # Placeholder: return empty result
        return GaussianResult(
            frame_id=frame_set.frame_id,
            num_gaussians=0,
            positions=np.empty((0, 3), dtype=np.float32),
            scales=np.empty((0, 3), dtype=np.float32),
            rotations=np.empty((0, 4), dtype=np.float32),
            opacities=np.empty(0, dtype=np.float32),
            sh_coeffs=np.empty((0, 48), dtype=np.float32),
            processing_time_ms=(time.time() - t_start) * 1000,
        )

    def shutdown(self):
        for ch, _ in self.channels:
            ch.close()
        self.channels.clear()


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_ply(path: str, result: GaussianResult):
    """Write Gaussians to PLY for visualization."""
    n = result.num_gaussians
    with open(path, "wb") as f:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
            "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
            "property float opacity\n"
        )
        for i in range(48):
            header += f"property float sh_{i}\n"
        header += "end_header\n"
        f.write(header.encode())

        if n > 0:
            # Pack into structured array
            dtype = np.dtype([
                ("pos", np.float32, 3),
                ("scale", np.float32, 3),
                ("rot", np.float32, 4),
                ("opacity", np.float32, 1),
                ("sh", np.float32, 48),
            ])
            data = np.empty(n, dtype=dtype)
            data["pos"] = result.positions
            data["scale"] = result.scales
            data["rot"] = result.rotations
            data["opacity"] = result.opacities.reshape(-1, 1)
            data["sh"] = result.sh_coeffs
            f.write(data.tobytes())


def write_json_stats(path: str, stats: List[dict]):
    """Write per-frame processing stats."""
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


# ---------------------------------------------------------------------------
# Calibration loader
# ---------------------------------------------------------------------------

def load_calibration(path: str) -> List[CameraCalibration]:
    """Load rig calibration from heimdall JSON schema format."""
    with open(path) as f:
        rig = json.load(f)

    calibrations = []
    for i, cam in enumerate(rig["cameras"]):
        K = np.array(cam["camera_matrix"], dtype=np.float32)

        rvec = np.array(cam["rvec"], dtype=np.float64)
        tvec = np.array(cam["tvec"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R.astype(np.float32)
        E[:3, 3] = tvec.astype(np.float32)

        dist = np.array(cam.get("dist_coeffs", [0, 0, 0, 0, 0]), dtype=np.float32)

        calibrations.append(CameraCalibration(
            index=cam.get("camera_index", i),
            serial_number=cam["serial_number"],
            width=cam["image_size"][0],
            height=cam["image_size"][1],
            intrinsics=K,
            extrinsics=E,
            dist_coeffs=dist,
        ))

    return calibrations


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_harness(args):
    print("=" * 60)
    print("Heimdall Test Harness")
    print("=" * 60)

    # Load calibration
    print(f"\nCalibration: {args.calibration}")
    calibrations = load_calibration(args.calibration)
    cal_dict = {c.index: c for c in calibrations}
    print(f"  {len(calibrations)} cameras loaded")

    # Open camera streams
    print(f"\nData: {args.data}")
    reader = MultiCameraReader(args.data, calibrations)

    # Initialize backend
    print(f"\nBackend: {args.backend}")
    if args.backend == "pytorch":
        backend = PyTorchBackend(
            checkpoint=args.checkpoint or "",
            device=f"cuda:{args.gpu}",
        )
    elif args.backend == "tensorrt":
        # TRT backend would instantiate the C++ TrtGaussianInference via pybind
        print("TensorRT backend not yet implemented in harness — use pytorch for now")
        sys.exit(1)
    elif args.backend == "runpod":
        workers = args.workers.split(",") if args.workers else []
        backend = RunPodBackend(workers)
    else:
        print(f"Unknown backend: {args.backend}")
        sys.exit(1)

    backend.initialize(
        calibrations,
        matting_checkpoint=args.matting_checkpoint or "",
    )
    print(f"  Backend ready: {backend.name}")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Process frames
    start_frame = args.start_frame
    end_frame = min(args.end_frame, len(reader)) if args.end_frame > 0 else len(reader)
    step = args.frame_step

    print(f"\nProcessing frames {start_frame}-{end_frame} (step {step})")
    print("-" * 60)

    stats = []
    total_gaussians = 0
    total_time = 0.0

    for frame_id in range(start_frame, end_frame, step):
        # Read synced frame set
        frame_set = reader.read_frame(frame_id)
        num_cams = len(frame_set.frames)

        if num_cams == 0:
            print(f"  Frame {frame_id}: no cameras available, skipping")
            continue

        # Run inference
        result = backend.process_frame(frame_set, cal_dict)

        # Write output
        if result.num_gaussians > 0 and args.write_ply:
            ply_path = os.path.join(args.output, f"frame_{frame_id:06d}.ply")
            write_ply(ply_path, result)

        # Stats
        total_gaussians += result.num_gaussians
        total_time += result.processing_time_ms

        frame_stats = {
            "frame_id": frame_id,
            "num_cameras": num_cams,
            "num_gaussians": result.num_gaussians,
            "processing_time_ms": round(result.processing_time_ms, 1),
        }
        stats.append(frame_stats)

        fps_est = 1000.0 / result.processing_time_ms if result.processing_time_ms > 0 else 0
        print(f"  Frame {frame_id:6d} | "
              f"{num_cams:2d} cams | "
              f"{result.num_gaussians:7d} gs | "
              f"{result.processing_time_ms:8.1f} ms | "
              f"{fps_est:5.1f} fps")

    # Summary
    num_frames = len(stats)
    print("-" * 60)
    if num_frames > 0:
        avg_time = total_time / num_frames
        avg_gs = total_gaussians / num_frames
        print(f"Frames processed: {num_frames}")
        print(f"Avg Gaussians/frame: {avg_gs:.0f}")
        print(f"Avg time/frame: {avg_time:.1f} ms ({1000/avg_time:.1f} fps)")
        print(f"Total time: {total_time/1000:.1f} s")
    else:
        print("No frames processed")

    # Write stats
    stats_path = os.path.join(args.output, "harness_stats.json")
    write_json_stats(stats_path, stats)
    print(f"\nStats written to: {stats_path}")

    # Cleanup
    reader.close()
    backend.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Heimdall Test Harness — replay multi-camera captures through Gaussian pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # PyTorch on M40 with real data
  %(prog)s --data /mnt/captures/session_01 --calibration rig.json --backend pytorch --gpu 0

  # PyTorch with specific model weights
  %(prog)s --data ./capture --calibration rig.json --backend pytorch --checkpoint model.pth

  # RunPod workers
  %(prog)s --data ./capture --calibration rig.json --backend runpod --workers w1:50051,w2:50051

  # Process every 10th frame (faster iteration)
  %(prog)s --data ./capture --calibration rig.json --backend pytorch --frame-step 10
        """,
    )

    # Data
    parser.add_argument("--data", required=True, help="Path to capture data directory")
    parser.add_argument("--calibration", required=True, help="Path to rig.json calibration")

    # Backend
    parser.add_argument("--backend", choices=["pytorch", "tensorrt", "runpod"],
                        default="pytorch", help="Inference backend")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device (pytorch/tensorrt)")
    parser.add_argument("--checkpoint", help="Model checkpoint path (pytorch)")
    parser.add_argument("--matting-checkpoint", help="Matting model checkpoint (pytorch)")
    parser.add_argument("--engine", help="TensorRT engine path (tensorrt)")
    parser.add_argument("--workers", help="Comma-separated worker addresses (runpod)")

    # Frame range
    parser.add_argument("--start-frame", type=int, default=0, help="First frame to process")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last frame (-1 = all)")
    parser.add_argument("--frame-step", type=int, default=1, help="Process every Nth frame")

    # Output
    parser.add_argument("--output", default="harness_output", help="Output directory")
    parser.add_argument("--write-ply", action="store_true", default=True,
                        help="Write PLY files per frame")
    parser.add_argument("--no-ply", dest="write_ply", action="store_false",
                        help="Skip PLY writing (stats only)")

    args = parser.parse_args()
    run_harness(args)


if __name__ == "__main__":
    main()
