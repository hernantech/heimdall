#!/usr/bin/env python3
"""
Evaluate a fine-tuned feed-forward Gaussian splatting model on held-out
validation data from a Heimdall volumetric capture rig.

Computes:
  - PSNR (peak signal-to-noise ratio)
  - SSIM (structural similarity)
  - LPIPS (learned perceptual image patch similarity)
  - Temporal consistency (optical flow magnitude between consecutive rendered frames)

Outputs a JSON report and optionally saves rendered images for visual inspection.

Usage:
    python evaluate.py \\
        --config config/finetune_config.yaml \\
        --checkpoint outputs/checkpoints/best.pth \\
        --output-dir outputs/eval_results

    # Evaluate on a specific split:
    python evaluate.py \\
        --config config/finetune_config.yaml \\
        --checkpoint outputs/checkpoints/best.pth \\
        --split val \\
        --save-images
"""

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml

# Re-use dataset and model loading from the fine-tuning script.
# Import paths are set up so that scripts/ is the package root.
from finetune import (
    GaussianFinetuneDataset,
    forward_model,
    load_config,
    load_model,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between two images [3, H, W] in [0, 1]."""
    mse = F.mse_loss(pred, target).item()
    if mse < 1e-10:
        return 100.0
    return -10.0 * np.log10(mse)


def compute_ssim(pred: torch.Tensor, target: torch.Tensor,
                 window_size: int = 11) -> float:
    """Compute SSIM between two images [3, H, W] in [0, 1].

    Returns the mean SSIM value (higher is better, max 1.0).
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    pred = pred.unsqueeze(0)    # [1, 3, H, W]
    target = target.unsqueeze(0)

    # Gaussian window.
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)
    window = window.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)

    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=3)
    mu_target = F.conv2d(target, window, padding=pad, groups=3)

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=3) - mu_pred ** 2
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=3) - mu_target ** 2
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=3) - mu_pred * mu_target

    ssim_map = ((2 * mu_pred * mu_target + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred ** 2 + mu_target ** 2 + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return ssim_map.mean().item()


class LPIPSMetric:
    """Wrapper around the lpips package for perceptual similarity."""

    def __init__(self, net: str = "vgg", device: torch.device = torch.device("cpu")):
        import lpips as lpips_module
        self.fn = lpips_module.LPIPS(net=net).to(device)
        self.fn.eval()
        for p in self.fn.parameters():
            p.requires_grad = False
        self.device = device

    @torch.no_grad()
    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS between two images [3, H, W] in [0, 1]. Lower is better."""
        # LPIPS expects [-1, 1].
        pred = (pred.unsqueeze(0) * 2 - 1).to(self.device)
        target = (target.unsqueeze(0) * 2 - 1).to(self.device)
        return self.fn(pred, target).item()


# ---------------------------------------------------------------------------
# Temporal consistency via optical flow
# ---------------------------------------------------------------------------

def compute_temporal_consistency(frames: list[np.ndarray]) -> dict[str, float]:
    """Compute temporal consistency metrics from a sequence of rendered frames.

    Uses Farneback optical flow between consecutive frames and reports
    the average and maximum flow magnitude. Lower values indicate smoother
    temporal transitions.

    Args:
        frames: List of rendered frames as uint8 BGR numpy arrays (OpenCV format).

    Returns:
        Dict with "mean_flow_magnitude", "max_flow_magnitude", "std_flow_magnitude".
    """
    if len(frames) < 2:
        return {"mean_flow_magnitude": 0.0, "max_flow_magnitude": 0.0,
                "std_flow_magnitude": 0.0}

    flow_magnitudes = []

    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray,
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        flow_magnitudes.append(float(magnitude.mean()))
        prev_gray = curr_gray

    return {
        "mean_flow_magnitude": float(np.mean(flow_magnitudes)),
        "max_flow_magnitude": float(np.max(flow_magnitudes)),
        "std_flow_magnitude": float(np.std(flow_magnitudes)),
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(cfg: dict, checkpoint_path: str, output_dir: Path,
             split: str = "val", save_images: bool = False) -> dict:
    """Run full evaluation and return metrics report."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    resolution = tuple(cfg["model"]["resolution"])
    num_source_views = cfg["data"]["num_source_views"]
    model_type = cfg["model"]["type"]

    # Load model.
    model = load_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    # Dataset.
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    dataset = GaussianFinetuneDataset(
        split_dir=str(dataset_dir / split),
        num_source_views=num_source_views,
        resolution=resolution,
        augmentations=None,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # LPIPS metric.
    lpips_net = cfg["loss"].get("lpips_net", "vgg")
    lpips_metric = LPIPSMetric(net=lpips_net, device=device)

    # Output directories.
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        (output_dir / "rendered").mkdir(exist_ok=True)
        (output_dir / "ground_truth").mkdir(exist_ok=True)
        (output_dir / "diff").mkdir(exist_ok=True)

    # Evaluate each sample.
    per_sample_metrics = []
    rendered_frames_for_temporal = []
    total_inference_time = 0.0

    for idx, batch in enumerate(loader):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        t0 = time.time()
        output = forward_model(model, batch, model_type)
        torch.cuda.synchronize() if device.type == "cuda" else None
        inference_time = time.time() - t0
        total_inference_time += inference_time

        rendered = output["rendered_image"][0].cpu()    # [3, H, W]
        target = batch["target_image"][0].cpu()         # [3, H, W]

        # Clamp to valid range.
        rendered = torch.clamp(rendered, 0.0, 1.0)

        # Compute metrics.
        psnr = compute_psnr(rendered, target)
        ssim = compute_ssim(rendered, target)
        lpips_val = lpips_metric.compute(rendered, target)

        sample_metrics = {
            "index": idx,
            "psnr": round(psnr, 4),
            "ssim": round(ssim, 4),
            "lpips": round(lpips_val, 4),
            "inference_time_ms": round(inference_time * 1000, 2),
        }
        per_sample_metrics.append(sample_metrics)

        # Save images for visual inspection.
        if save_images:
            rendered_bgr = _tensor_to_bgr(rendered)
            target_bgr = _tensor_to_bgr(target)

            cv2.imwrite(str(output_dir / "rendered" / f"{idx:06d}.png"), rendered_bgr)
            cv2.imwrite(str(output_dir / "ground_truth" / f"{idx:06d}.png"), target_bgr)

            # Error heatmap (absolute difference, amplified for visibility).
            diff = np.abs(rendered_bgr.astype(np.float32) - target_bgr.astype(np.float32))
            diff = np.clip(diff * 5, 0, 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / "diff" / f"{idx:06d}.png"), diff)

        # Collect rendered frames for temporal consistency.
        rendered_bgr = _tensor_to_bgr(rendered)
        rendered_frames_for_temporal.append(rendered_bgr)

        if (idx + 1) % 50 == 0:
            logger.info("  Evaluated %d / %d samples", idx + 1, len(dataset))

    # Aggregate metrics.
    psnr_values = [m["psnr"] for m in per_sample_metrics]
    ssim_values = [m["ssim"] for m in per_sample_metrics]
    lpips_values = [m["lpips"] for m in per_sample_metrics]

    aggregate = {
        "psnr": {
            "mean": round(float(np.mean(psnr_values)), 4),
            "std": round(float(np.std(psnr_values)), 4),
            "min": round(float(np.min(psnr_values)), 4),
            "max": round(float(np.max(psnr_values)), 4),
        },
        "ssim": {
            "mean": round(float(np.mean(ssim_values)), 4),
            "std": round(float(np.std(ssim_values)), 4),
            "min": round(float(np.min(ssim_values)), 4),
            "max": round(float(np.max(ssim_values)), 4),
        },
        "lpips": {
            "mean": round(float(np.mean(lpips_values)), 4),
            "std": round(float(np.std(lpips_values)), 4),
            "min": round(float(np.min(lpips_values)), 4),
            "max": round(float(np.max(lpips_values)), 4),
        },
    }

    # Temporal consistency (only meaningful if samples are from a temporal sequence).
    temporal = compute_temporal_consistency(rendered_frames_for_temporal)

    # Inference throughput.
    n_samples = len(per_sample_metrics)
    avg_inference_ms = (total_inference_time / max(1, n_samples)) * 1000

    report = {
        "checkpoint": checkpoint_path,
        "split": split,
        "model_type": model_type,
        "resolution": list(resolution),
        "num_source_views": num_source_views,
        "num_samples": n_samples,
        "aggregate_metrics": aggregate,
        "temporal_consistency": temporal,
        "inference": {
            "avg_time_ms": round(avg_inference_ms, 2),
            "total_time_s": round(total_inference_time, 2),
        },
        "per_sample_metrics": per_sample_metrics,
    }

    # Write report.
    report_path = output_dir / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary.
    logger.info("=" * 60)
    logger.info("Evaluation Report  (%s, %d samples)", split, n_samples)
    logger.info("=" * 60)
    logger.info("  PSNR:   %.2f +/- %.2f  (range: %.2f - %.2f)",
                aggregate["psnr"]["mean"], aggregate["psnr"]["std"],
                aggregate["psnr"]["min"], aggregate["psnr"]["max"])
    logger.info("  SSIM:   %.4f +/- %.4f  (range: %.4f - %.4f)",
                aggregate["ssim"]["mean"], aggregate["ssim"]["std"],
                aggregate["ssim"]["min"], aggregate["ssim"]["max"])
    logger.info("  LPIPS:  %.4f +/- %.4f  (range: %.4f - %.4f)",
                aggregate["lpips"]["mean"], aggregate["lpips"]["std"],
                aggregate["lpips"]["min"], aggregate["lpips"]["max"])
    logger.info("  Temporal flow (mean): %.4f  (std: %.4f, max: %.4f)",
                temporal["mean_flow_magnitude"],
                temporal["std_flow_magnitude"],
                temporal["max_flow_magnitude"])
    logger.info("  Inference: %.1f ms/sample avg", avg_inference_ms)
    logger.info("  Report saved to: %s", report_path)

    return report


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert a [3, H, W] float tensor in [0, 1] to uint8 BGR numpy array."""
    img = tensor.permute(1, 2, 0).numpy()  # [H, W, 3] RGB
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned Gaussian splatting model on Heimdall rig data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to finetune_config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--output-dir", default="outputs/eval_results",
                        help="Directory for evaluation outputs")
    parser.add_argument("--split", default="val", choices=["train", "val"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--save-images", action="store_true",
                        help="Save rendered images, ground truth, and diff maps")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)

    report = evaluate(
        cfg=cfg,
        checkpoint_path=args.checkpoint,
        output_dir=Path(args.output_dir),
        split=args.split,
        save_images=args.save_images,
    )

    # Exit with non-zero code if quality is below threshold (useful in CI).
    mean_psnr = report["aggregate_metrics"]["psnr"]["mean"]
    if mean_psnr < 20.0:
        logger.warning("PSNR %.2f is below 20 dB threshold — model may need more training", mean_psnr)


if __name__ == "__main__":
    main()
