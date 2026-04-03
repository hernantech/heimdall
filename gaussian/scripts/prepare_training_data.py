#!/usr/bin/env python3
"""
Prepare training data for fine-tuning a feed-forward Gaussian splatting model
(DepthSplat/MVSplat) on a Heimdall volumetric capture rig.

Reads captured multi-view sequences (JPEG or OpenEXR) with known camera
calibration and produces a standardized dataset directory:

    dataset/
      train/
        sequence_000/
          images/          (K source + 1 target as .png)
          cameras.json     (intrinsics + extrinsics per image)
      val/
        ... (same structure, different sequences or held-out frames)

Each training sample consists of K source views and 1 held-out target view.
The model learns to predict the target from the sources.

Usage:
    python prepare_training_data.py \\
        --input-dir /capture/session_2026_03_15 \\
        --calibration /calibration/rig_v3.json \\
        --output-dir /data/heimdall/training_dataset \\
        --num-source-views 6 \\
        --val-ratio 0.15 \\
        --frame-step 5 \\
        --resolution 512 960

Input directory structure (expected):
    <input-dir>/
      cam_00/          (or by serial number: cam_12345678/)
        frame_000000.jpg   (or .exr)
        frame_000001.jpg
        ...
      cam_01/
        ...
"""

import argparse
import json
import logging
import random
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def load_calibration(calibration_path: str) -> dict:
    """Load a Heimdall rig calibration file (rig.schema.json format).

    Returns a dict mapping camera_index -> camera info dict with keys:
        serial_number, image_size, camera_matrix (3x3 np), dist_coeffs (np),
        rvec (np), tvec (np), extrinsic_4x4 (np 4x4 world-to-camera).
    """
    with open(calibration_path, "r") as f:
        rig = json.load(f)

    if rig.get("version") != "1.0":
        logger.warning("Unexpected rig schema version: %s", rig.get("version"))

    cameras = {}
    for i, cam in enumerate(rig["cameras"]):
        idx = cam.get("camera_index", i)
        K = np.array(cam["camera_matrix"], dtype=np.float64)       # 3x3
        dist = np.array(cam["dist_coeffs"], dtype=np.float64)
        rvec = np.array(cam["rvec"], dtype=np.float64)             # 3,
        tvec = np.array(cam["tvec"], dtype=np.float64)             # 3,

        # Build 4x4 extrinsic matrix.
        if "rotation_matrix" in cam:
            R = np.array(cam["rotation_matrix"], dtype=np.float64)
        else:
            R, _ = cv2.Rodrigues(rvec)

        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = tvec

        cameras[idx] = {
            "serial_number": cam["serial_number"],
            "image_size": cam["image_size"],       # [width, height]
            "camera_matrix": K,
            "dist_coeffs": dist,
            "rvec": rvec,
            "tvec": tvec,
            "extrinsic_4x4": extrinsic,
        }

    logger.info("Loaded calibration for %d cameras (rig_id=%s)", len(cameras), rig.get("rig_id"))
    return cameras


# ---------------------------------------------------------------------------
# Frame discovery
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff"}


def discover_frames(input_dir: Path) -> dict[int, list[Path]]:
    """Discover per-camera frame files from the capture directory.

    Expects subdirectories named cam_XX or cam_<serial>.  Returns a dict
    mapping camera_index -> sorted list of frame file paths.
    """
    cam_dirs = sorted(input_dir.iterdir())
    frames_by_cam: dict[int, list[Path]] = {}

    for cam_dir in cam_dirs:
        if not cam_dir.is_dir():
            continue
        name = cam_dir.name
        # Parse camera index from directory name (cam_00, cam_01, ...).
        # Fallback: try to match by serial number later.
        if name.startswith("cam_"):
            try:
                cam_idx = int(name.split("_", 1)[1])
            except ValueError:
                logger.warning("Cannot parse camera index from %s, skipping", name)
                continue
        else:
            continue

        frame_files = sorted(
            f for f in cam_dir.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if frame_files:
            frames_by_cam[cam_idx] = frame_files
            logger.debug("Camera %d: %d frames in %s", cam_idx, len(frame_files), cam_dir)

    logger.info("Discovered %d cameras with frames", len(frames_by_cam))
    return frames_by_cam


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    """Load an image as uint8 BGR (OpenCV convention).

    Supports JPEG, PNG, TIFF, and OpenEXR (tone-mapped to 8-bit).
    """
    suffix = path.suffix.lower()

    if suffix == ".exr":
        # OpenEXR: load as float, tone-map to 8-bit for training.
        img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise FileNotFoundError(f"Cannot read EXR: {path}")
        # Simple Reinhard tone-mapping.
        img = np.clip(img, 0, None)
        img = img / (1.0 + img)
        img = (img * 255).astype(np.uint8)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

    return img


def undistort_image(img: np.ndarray, K: np.ndarray, dist: np.ndarray,
                    target_size: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Undistort an image using OpenCV calibration parameters.

    Returns (undistorted_image, new_camera_matrix).  If target_size is given
    the image is also resized.
    """
    h, w = img.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0)
    undistorted = cv2.undistort(img, K, dist, None, new_K)

    # Crop to valid region.
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y:y + rh, x:x + rw]
        # Adjust principal point for the crop.
        new_K[0, 2] -= x
        new_K[1, 2] -= y

    if target_size is not None:
        th, tw = target_size
        ch, cw = undistorted.shape[:2]
        # Scale intrinsics to match the resized image.
        sx, sy = tw / cw, th / ch
        new_K[0, :] *= sx
        new_K[1, :] *= sy
        undistorted = cv2.resize(undistorted, (tw, th), interpolation=cv2.INTER_AREA)

    return undistorted, new_K


# ---------------------------------------------------------------------------
# Source-view selection strategies
# ---------------------------------------------------------------------------

def select_source_views_spatial(target_idx: int, all_indices: list[int],
                                cameras: dict, k: int) -> list[int]:
    """Select K source views that are well-distributed around the target.

    Uses angular distance between camera positions relative to the scene
    center to pick views that provide diverse coverage.
    """
    # Camera positions in world frame: C = -R^T @ t
    positions = {}
    for idx in all_indices:
        R = cameras[idx]["extrinsic_4x4"][:3, :3]
        t = cameras[idx]["extrinsic_4x4"][:3, 3]
        positions[idx] = -R.T @ t

    candidates = [i for i in all_indices if i != target_idx]
    if len(candidates) <= k:
        return candidates

    target_pos = positions[target_idx]

    # Sort by distance to target (nearest first), then greedily pick
    # views that are maximally spread from already-selected views.
    candidates.sort(key=lambda i: np.linalg.norm(positions[i] - target_pos))

    selected = [candidates[0]]
    remaining = set(candidates[1:])

    while len(selected) < k and remaining:
        best = None
        best_score = -1.0
        for cand in remaining:
            # Score = minimum angular separation from any already-selected view,
            # seen from the scene center (origin).
            cand_dir = positions[cand] / (np.linalg.norm(positions[cand]) + 1e-8)
            min_angle = float("inf")
            for s in selected:
                s_dir = positions[s] / (np.linalg.norm(positions[s]) + 1e-8)
                angle = np.arccos(np.clip(np.dot(cand_dir, s_dir), -1, 1))
                min_angle = min(min_angle, angle)
            if min_angle > best_score:
                best_score = min_angle
                best = cand
        selected.append(best)
        remaining.discard(best)

    return selected


def select_source_views_random(target_idx: int, all_indices: list[int],
                               k: int, rng: random.Random) -> list[int]:
    """Randomly select K source views (excluding the target)."""
    candidates = [i for i in all_indices if i != target_idx]
    return rng.sample(candidates, min(k, len(candidates)))


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

def camera_to_json(K: np.ndarray, extrinsic: np.ndarray) -> dict:
    """Serialize a camera's intrinsics and extrinsic to JSON-friendly dicts."""
    return {
        "intrinsic": K.tolist(),                   # 3x3
        "extrinsic": extrinsic.tolist(),           # 4x4 world-to-camera
    }


def generate_sample(
    sample_idx: int,
    frame_idx: int,
    target_cam: int,
    source_cams: list[int],
    frames_by_cam: dict[int, list[Path]],
    cameras: dict,
    target_size: tuple[int, int] | None,
    output_dir: Path,
) -> dict[str, Any] | None:
    """Generate one training sample (K source views + 1 target view).

    Returns the cameras.json dict for this sample, or None on failure.
    """
    seq_name = f"sequence_{sample_idx:06d}"
    seq_dir = output_dir / seq_name
    img_dir = seq_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    cameras_json: dict[str, Any] = {
        "frame_index": frame_idx,
        "target_camera_index": target_cam,
        "source_camera_indices": source_cams,
        "views": {},
    }

    all_cams = source_cams + [target_cam]
    for cam_idx in all_cams:
        if frame_idx >= len(frames_by_cam[cam_idx]):
            logger.warning("Frame %d out of range for camera %d", frame_idx, cam_idx)
            return None

        frame_path = frames_by_cam[cam_idx][frame_idx]
        try:
            img = load_image(frame_path)
        except FileNotFoundError:
            logger.warning("Cannot load %s", frame_path)
            return None

        cal = cameras[cam_idx]

        # Undistort and resize.
        img, new_K = undistort_image(img, cal["camera_matrix"], cal["dist_coeffs"],
                                     target_size=target_size)

        # Determine role tag for filename.
        if cam_idx == target_cam:
            role = "target"
        else:
            src_rank = source_cams.index(cam_idx)
            role = f"source_{src_rank:02d}"

        filename = f"{role}_cam{cam_idx:02d}.png"
        cv2.imwrite(str(img_dir / filename), img)

        cameras_json["views"][filename] = camera_to_json(new_K, cal["extrinsic_4x4"])

    # Write cameras.json for this sample.
    with open(seq_dir / "cameras.json", "w") as f:
        json.dump(cameras_json, f, indent=2)

    return cameras_json


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data for Heimdall Gaussian splatting fine-tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Root directory of captured multi-view sequences")
    parser.add_argument("--calibration", required=True,
                        help="Path to rig calibration JSON (rig.schema.json format)")
    parser.add_argument("--output-dir", required=True,
                        help="Output dataset directory")
    parser.add_argument("--num-source-views", type=int, default=6,
                        help="Number of source views per training sample (K)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Fraction of frames reserved for validation")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="Use every N-th frame (subsampling)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[512, 960],
                        metavar=("H", "W"),
                        help="Target resolution [height, width]")
    parser.add_argument("--view-selection", choices=["spatial", "random"], default="spatial",
                        help="Strategy for selecting source views")
    parser.add_argument("--targets-per-frame", type=int, default=3,
                        help="Number of distinct target views to hold out per frame")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_size = tuple(args.resolution)  # (H, W)

    if not input_dir.is_dir():
        logger.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    # Load calibration and discover frames.
    cameras = load_calibration(args.calibration)
    frames_by_cam = discover_frames(input_dir)

    if not frames_by_cam:
        logger.error("No camera frames found in %s", input_dir)
        sys.exit(1)

    # Only use cameras that appear in both calibration and captures.
    common_cams = sorted(set(cameras.keys()) & set(frames_by_cam.keys()))
    if len(common_cams) < args.num_source_views + 1:
        logger.error(
            "Need at least %d cameras (K+1) but only %d have both calibration and frames",
            args.num_source_views + 1, len(common_cams),
        )
        sys.exit(1)

    logger.info("Using %d cameras: %s", len(common_cams), common_cams)

    # Determine frame count (use the minimum across all cameras).
    min_frames = min(len(frames_by_cam[c]) for c in common_cams)
    frame_indices = list(range(0, min_frames, args.frame_step))
    logger.info("Total frames available: %d, after subsampling (step=%d): %d",
                min_frames, args.frame_step, len(frame_indices))

    # Train/val split by frame index (temporal split — ensures val frames are
    # unseen during training, which is important for temporal generalization).
    rng = random.Random(args.seed)
    shuffled = frame_indices.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_ratio))
    val_frames = set(shuffled[:n_val])
    train_frames = set(shuffled[n_val:])
    logger.info("Train frames: %d, Val frames: %d", len(train_frames), len(val_frames))

    # Generate training samples.
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    sample_idx = 0
    stats = {"train": 0, "val": 0, "failed": 0}

    for frame_idx in sorted(frame_indices):
        split = "val" if frame_idx in val_frames else "train"
        split_dir = val_dir if split == "val" else train_dir

        # Select multiple target views per frame for data diversity.
        if args.view_selection == "random":
            target_cams = rng.sample(common_cams, min(args.targets_per_frame, len(common_cams)))
        else:
            # Spread target views evenly across the rig.
            step = max(1, len(common_cams) // args.targets_per_frame)
            target_cams = common_cams[::step][:args.targets_per_frame]

        for target_cam in target_cams:
            # Select source views.
            if args.view_selection == "spatial":
                source_cams = select_source_views_spatial(
                    target_cam, common_cams, cameras, args.num_source_views)
            else:
                source_cams = select_source_views_random(
                    target_cam, common_cams, args.num_source_views, rng)

            result = generate_sample(
                sample_idx=sample_idx,
                frame_idx=frame_idx,
                target_cam=target_cam,
                source_cams=source_cams,
                frames_by_cam=frames_by_cam,
                cameras=cameras,
                target_size=target_size,
                output_dir=split_dir,
            )

            if result is not None:
                stats[split] += 1
                sample_idx += 1
            else:
                stats["failed"] += 1

    # Write dataset metadata.
    metadata = {
        "created_by": "heimdall prepare_training_data.py",
        "calibration": str(args.calibration),
        "input_dir": str(args.input_dir),
        "num_source_views": args.num_source_views,
        "resolution": list(target_size),
        "frame_step": args.frame_step,
        "val_ratio": args.val_ratio,
        "view_selection": args.view_selection,
        "num_cameras": len(common_cams),
        "camera_indices": common_cams,
        "stats": stats,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset generation complete.")
    logger.info("  Train samples: %d", stats["train"])
    logger.info("  Val samples:   %d", stats["val"])
    logger.info("  Failed:        %d", stats["failed"])
    logger.info("  Output:        %s", output_dir)


if __name__ == "__main__":
    main()
