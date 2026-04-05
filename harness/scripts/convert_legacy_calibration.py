#!/usr/bin/env python3
"""
Convert legacy camera calibration formats to heimdall rig.schema.json.

Supports two common legacy formats found in volumetric capture systems:

  Format A ("recorder-style"):
    - focal, focal_y (focal lengths in pixels)
    - k1_deg2, k2_deg4, k3_deg6 (radial distortion)
    - rotate_mat[9] or rotate[3] (rotation)
    - translate[3] (translation)
    - colTrans_mat[9], colTrans_vec[3] (color correction)
    - win_translate[2] (principal point offset)

  Format B ("solver-style"):
    - cx, cy (principal point)
    - x_focallen, y_focallen (focal lengths)
    - radial_poly[17] (17-coefficient distortion polynomial)
    - extrinsics.mtx[16] (4x4 world-to-camera matrix)
    - colour_transform[9] (3x3 color correction)

Usage:
    python convert_legacy_calibration.py \
        --input /path/to/camera_files/ \
        --output rig.json \
        --format auto
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


def parse_recorder_style(data: dict, idx: int) -> dict:
    """Parse recorder-style calibration (focal + k1/k2/k3 + rotation matrix)."""
    focal = float(data.get("focal", 0))
    focal_y = float(data.get("focal_y", focal))

    width = int(data.get("image_width", 1920))
    height = int(data.get("image_height", 1080))

    cx = width / 2.0
    cy = height / 2.0
    if "win_translate" in data:
        wt = data["win_translate"]
        if isinstance(wt, list) and len(wt) >= 2:
            cx += float(wt[0])
            cy += float(wt[1])

    K = [[focal, 0, cx], [0, focal_y, cy], [0, 0, 1]]

    k1 = float(data.get("k1_deg2", 0))
    k2 = float(data.get("k2_deg4", 0))
    k3 = float(data.get("k3_deg6", 0))
    dist = [k1, k2, 0, 0, k3]

    if "rotate_mat" in data:
        rm = [float(x) for x in data["rotate_mat"]]
        R = np.array(rm).reshape(3, 3)
    elif "rotate" in data:
        rvec = np.array([float(x) for x in data["rotate"]])
        import cv2
        R, _ = cv2.Rodrigues(rvec)
    else:
        R = np.eye(3)

    t = np.array([float(x) for x in data.get("translate", [0, 0, 0])])

    import cv2
    rvec, _ = cv2.Rodrigues(R)

    color_matrix = None
    if "colTrans_mat" in data:
        cm = [float(x) for x in data["colTrans_mat"]]
        color_matrix = [cm[0:3], cm[3:6], cm[6:9]]

    serial = int(data.get("serial_number", data.get("camera_index", idx)))

    cam = {
        "serial_number": serial,
        "camera_index": idx,
        "image_size": [width, height],
        "camera_matrix": K,
        "dist_coeffs": dist,
        "rvec": rvec.flatten().tolist(),
        "tvec": t.tolist(),
    }

    if color_matrix:
        cam["color_correction"] = {
            "matrix_3x3": color_matrix,
            "offset": [float(x) for x in data.get("colTrans_vec", [0, 0, 0])],
        }

    return cam


def parse_solver_style(data: dict, idx: int) -> dict:
    """Parse solver-style calibration (cx/cy + focal + radial_poly + 4x4 extrinsics)."""
    cx = float(data.get("cx", 0))
    cy = float(data.get("cy", 0))
    fx = float(data.get("x_focallen", 0))
    fy = float(data.get("y_focallen", fx))

    K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    poly = data.get("radial_poly", [0] * 17)
    dist = [float(poly[1]) if len(poly) > 1 else 0,
            float(poly[2]) if len(poly) > 2 else 0,
            0, 0,
            float(poly[3]) if len(poly) > 3 else 0]

    ext = data.get("extrinsics", {})
    mtx = ext.get("mtx", list(np.eye(4).flatten()))
    E = np.array([float(x) for x in mtx]).reshape(4, 4)

    R = E[:3, :3]
    t = E[:3, 3]

    import cv2
    rvec, _ = cv2.Rodrigues(R)

    serial = int(data.get("serial_number", idx))
    width = int(data.get("width", data.get("image_width", 1920)))
    height = int(data.get("height", data.get("image_height", 1080)))

    cam = {
        "serial_number": serial,
        "camera_index": idx,
        "image_size": [width, height],
        "camera_matrix": K,
        "dist_coeffs": dist,
        "rvec": rvec.flatten().tolist(),
        "tvec": t.tolist(),
    }

    if "colour_transform" in data:
        ct = [float(x) for x in data["colour_transform"]]
        cam["color_correction"] = {
            "matrix_3x3": [ct[0:3], ct[3:6], ct[6:9]],
            "offset": [0, 0, 0],
        }

    return cam


def detect_format(data: dict) -> str:
    """Auto-detect calibration format from field names."""
    if "x_focallen" in data or "radial_poly" in data:
        return "solver"
    return "recorder"


def main():
    parser = argparse.ArgumentParser(description="Convert legacy calibration to heimdall format")
    parser.add_argument("--input", required=True, help="Path to camera files (directory or single file)")
    parser.add_argument("--output", default="rig.json", help="Output rig.json path")
    parser.add_argument("--format", choices=["recorder", "solver", "auto"], default="auto",
                        help="Legacy calibration format")
    parser.add_argument("--rig-id", default="converted", help="Rig identifier")
    args = parser.parse_args()

    input_path = args.input
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.json")))
        files = [f for f in files if "camera" in os.path.basename(f).lower()
                 or "cam" in os.path.basename(f).lower()]
        if not files:
            files = sorted(glob.glob(os.path.join(input_path, "*.json")))
    else:
        files = [input_path]

    if not files:
        print(f"No calibration files found in {input_path}")
        sys.exit(1)

    print(f"Found {len(files)} calibration files")

    cameras = []
    for i, fpath in enumerate(files):
        with open(fpath) as f:
            data = json.load(f)

        fmt = args.format if args.format != "auto" else detect_format(data)

        if fmt == "recorder":
            cam = parse_recorder_style(data, i)
        else:
            cam = parse_solver_style(data, i)

        cameras.append(cam)
        print(f"  {os.path.basename(fpath)} → camera {cam['camera_index']} "
              f"(serial {cam['serial_number']}, {cam['image_size'][0]}x{cam['image_size'][1]})")

    rig = {
        "version": "1.0",
        "rig_id": args.rig_id,
        "cameras": cameras,
    }

    with open(args.output, "w") as f:
        json.dump(rig, f, indent=2)

    print(f"\nWritten {len(cameras)} cameras to {args.output}")


if __name__ == "__main__":
    main()
