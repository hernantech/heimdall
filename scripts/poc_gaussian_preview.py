#!/usr/bin/env python3
"""
Heimdall PoC: Single-frame Gaussian splatting preview.

Runs on Tesla M40 (sm_52, 24GB) using PyTorch directly (no TensorRT).
Processes one frame: load images → matting → GPS-Gaussian → output splats.

Usage:
    python scripts/poc_gaussian_preview.py \
        --images /path/to/frame/cam_*.jpg \
        --calibration /path/to/rig.json \
        --output /path/to/output.ply \
        --gpu 0

For testing without real capture data:
    python scripts/poc_gaussian_preview.py --synthetic --gpu 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch


def check_gpu():
    """Verify GPU is available and print info."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        sys.exit(1)

    device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(device)
    cap = torch.cuda.get_device_capability(device)
    mem = torch.cuda.get_device_properties(device).total_mem / 1e9
    print(f"GPU: {name} (sm_{cap[0]}{cap[1]}, {mem:.1f} GB)")

    if cap < (5, 2):
        print("ERROR: Compute capability too low (need >= 5.2)")
        sys.exit(1)

    return device


def load_calibration(path):
    """Load rig calibration from heimdall JSON schema format."""
    with open(path) as f:
        rig = json.load(f)

    cameras = []
    for cam in rig["cameras"]:
        K = np.array(cam["camera_matrix"], dtype=np.float32)  # 3x3
        # Build 4x4 extrinsic from rvec + tvec
        import cv2
        rvec = np.array(cam["rvec"], dtype=np.float64)
        tvec = np.array(cam["tvec"], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R.astype(np.float32)
        E[:3, 3] = tvec.astype(np.float32)
        cameras.append({
            "serial": cam["serial_number"],
            "width": cam["image_size"][0],
            "height": cam["image_size"][1],
            "intrinsics": K,
            "extrinsics": E,
        })
    return cameras


def generate_synthetic_data(num_cameras=8, resolution=512):
    """Generate synthetic multi-view data for testing without real cameras."""
    print(f"Generating synthetic data: {num_cameras} cameras, {resolution}x{resolution}")

    cameras = []
    images = []

    for i in range(num_cameras):
        angle = 2 * np.pi * i / num_cameras
        radius = 2.0  # meters from center

        # Camera position on a circle
        pos = np.array([radius * np.cos(angle), 1.2, radius * np.sin(angle)])
        # Look at origin
        forward = -pos / np.linalg.norm(pos)
        up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        R = np.stack([right, -up, forward], axis=0)  # world-to-camera rotation
        t = -R @ pos

        E = np.eye(4, dtype=np.float32)
        E[:3, :3] = R.astype(np.float32)
        E[:3, 3] = t.astype(np.float32)

        fx = fy = resolution * 1.2
        cx, cy = resolution / 2, resolution / 2
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        cameras.append({
            "serial": 10000 + i,
            "width": resolution,
            "height": resolution,
            "intrinsics": K,
            "extrinsics": E,
        })

        # Synthetic image: gradient + noise (not meaningful, just for pipeline testing)
        img = np.random.randint(0, 255, (resolution, resolution, 3), dtype=np.uint8)
        images.append(img)

    return cameras, images


def load_images(image_paths):
    """Load images from disk."""
    from PIL import Image
    images = []
    for p in sorted(image_paths):
        img = np.array(Image.open(p).convert("RGB"))
        images.append(img)
    return images


def run_matting(images, device):
    """
    Run background matting on images.
    Uses RVM (Robust Video Matting) in PyTorch — no TensorRT needed.
    Falls back to simple thresholding if RVM not available.
    """
    try:
        # Try loading RVM
        sys.path.insert(0, "third_party/RobustVideoMatting")
        from model import MattingNetwork
        model = MattingNetwork("mobilenetv3").eval().to(device)
        # Load pretrained weights if available
        weights_path = "third_party/RobustVideoMatting/rvm_mobilenetv3.pth"
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
            print("  RVM loaded (MobileNetV3)")
        else:
            print("  RVM weights not found — using random init (output will be meaningless)")

        masks = []
        rec = [None] * 4  # RVM recurrent states
        with torch.no_grad():
            for img in images:
                # Normalize and resize
                x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                x = torch.nn.functional.interpolate(x, size=(288, 512), mode="bilinear", align_corners=False)
                fgr, pha, *rec = model(x, *rec, downsample_ratio=0.25)
                mask = pha.squeeze().cpu().numpy()
                masks.append(mask)
        return masks

    except Exception as e:
        print(f"  RVM not available ({e}) — using dummy masks (all foreground)")
        return [np.ones((images[0].shape[0], images[0].shape[1]), dtype=np.float32)] * len(images)


def run_gps_gaussian(images, cameras, masks, device):
    """
    Run GPS-Gaussian inference on stereo pairs.
    Uses PyTorch directly — no TensorRT needed.
    Falls back to depth-based unprojection if GPS-Gaussian not available.
    """
    try:
        sys.path.insert(0, "third_party/GPS-Gaussian")
        # GPS-Gaussian integration would go here
        # from gps_gaussian import GPSGaussianModel
        raise ImportError("GPS-Gaussian PyTorch integration not yet implemented")

    except Exception as e:
        print(f"  GPS-Gaussian not available ({e})")
        print("  Falling back to depth-based Gaussian generation (synthetic PoC)")

        # Fallback: generate Gaussians from random depth + known calibration
        # This validates the downstream pipeline (tracking, SPZ, streaming)
        # without requiring the actual GPS-Gaussian model.

        all_gaussians = []
        num_pairs = min(len(cameras) // 2, 4)

        for pair_idx in range(num_pairs):
            cam_a = cameras[pair_idx * 2]
            cam_b = cameras[pair_idx * 2 + 1]

            # Generate random points in a human-shaped volume
            num_points = 10000
            # Cylinder approximation of a human: radius 0.3m, height 1.8m
            theta = np.random.uniform(0, 2 * np.pi, num_points)
            r = np.random.uniform(0, 0.3, num_points)
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            y = np.random.uniform(0, 1.8, num_points)

            for i in range(num_points):
                g = {
                    "position": [float(x[i]), float(y[i]), float(z[i])],
                    "scale": [0.005, 0.005, 0.005],
                    "rotation": [1.0, 0.0, 0.0, 0.0],  # wxyz identity
                    "opacity": float(np.random.uniform(0.5, 1.0)),
                    "sh": [float(np.random.uniform(-1, 1)) for _ in range(48)],
                }
                all_gaussians.append(g)

            print(f"  Pair {pair_idx}: cameras {cam_a['serial']}/{cam_b['serial']} → {num_points} Gaussians")

        return all_gaussians


def write_ply(path, gaussians):
    """Write Gaussians to a PLY file for visualization."""
    n = len(gaussians)
    print(f"Writing {n} Gaussians to {path}")

    with open(path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float scale_0\n")
        f.write("property float scale_1\n")
        f.write("property float scale_2\n")
        f.write("property float rot_0\n")
        f.write("property float rot_1\n")
        f.write("property float rot_2\n")
        f.write("property float rot_3\n")
        f.write("property float opacity\n")
        for i in range(48):
            f.write(f"property float sh_{i}\n")
        f.write("end_header\n")

        for g in gaussians:
            pos = g["position"]
            sc = g["scale"]
            rot = g["rotation"]
            vals = [pos[0], pos[1], pos[2],
                    sc[0], sc[1], sc[2],
                    rot[0], rot[1], rot[2], rot[3],
                    g["opacity"]] + g["sh"]
            f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Heimdall PoC: Gaussian preview")
    parser.add_argument("--images", nargs="*", help="Camera image paths (glob)")
    parser.add_argument("--calibration", help="Path to rig.json calibration")
    parser.add_argument("--output", default="poc_output.ply", help="Output PLY path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test data")
    parser.add_argument("--num-cameras", type=int, default=8, help="Synthetic camera count")
    parser.add_argument("--resolution", type=int, default=512, help="Synthetic resolution")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = check_gpu()

    t_start = time.time()

    # Load or generate data
    if args.synthetic:
        cameras, images = generate_synthetic_data(args.num_cameras, args.resolution)
    else:
        if not args.calibration or not args.images:
            print("ERROR: --calibration and --images required (or use --synthetic)")
            sys.exit(1)
        cameras = load_calibration(args.calibration)
        images = load_images(args.images)

    print(f"Loaded {len(cameras)} cameras, {len(images)} images")

    # Step 1: Matting
    print("\n--- Matting ---")
    t_mat = time.time()
    masks = run_matting(images, f"cuda:{args.gpu}")
    print(f"  Matting: {time.time() - t_mat:.2f}s")

    # Step 2: GPS-Gaussian inference
    print("\n--- Gaussian Inference ---")
    t_gs = time.time()
    gaussians = run_gps_gaussian(images, cameras, masks, f"cuda:{args.gpu}")
    print(f"  Inference: {time.time() - t_gs:.2f}s")
    print(f"  Total Gaussians: {len(gaussians)}")

    # Step 3: Write output
    print("\n--- Output ---")
    write_ply(args.output, gaussians)

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.2f}s")
    print(f"Output: {args.output}")
    print(f"\nView with: python -c \"import open3d; pcd=open3d.io.read_point_cloud('{args.output}'); open3d.visualization.draw_geometries([pcd])\"")


if __name__ == "__main__":
    main()
