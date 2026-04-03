#!/usr/bin/env python3
"""
Fine-tune a feed-forward Gaussian splatting model (DepthSplat/MVSplat)
on Heimdall volumetric capture rig data.

This is a clean PyTorch training loop that:
  - Loads a pretrained DepthSplat or MVSplat checkpoint
  - Fine-tunes on data prepared by prepare_training_data.py
  - Uses photometric loss (L1 + SSIM) + optional LPIPS on rendered target views
  - Supports learning rate warmup, cosine decay, gradient clipping
  - Saves checkpoints periodically and logs to TensorBoard or W&B

Usage:
    python finetune.py --config config/finetune_config.yaml

    # Override specific settings:
    python finetune.py --config config/finetune_config.yaml \\
        --training.lr 5e-5 --training.epochs 50

Requires: torch, torchvision, lpips, pyyaml, tensorboard (or wandb).
Assumes depthsplat / mvsplat are installed as importable Python packages.
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """Load YAML config and apply dot-notation CLI overrides.

    Overrides are key=value strings like "training.lr=5e-5".
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if overrides:
        for ov in overrides:
            if "=" not in ov:
                continue
            key, val = ov.split("=", 1)
            keys = key.split(".")
            d = cfg
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            # Auto-cast value.
            try:
                val = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                pass
            d[keys[-1]] = val

    return cfg


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GaussianFinetuneDataset(Dataset):
    """Dataset of (source_images, source_cameras, target_image, target_camera) tuples.

    Reads the directory structure produced by prepare_training_data.py:
        split_dir/
          sequence_NNNNNN/
            images/
              source_00_camXX.png
              source_01_camXX.png
              ...
              target_camXX.png
            cameras.json
    """

    def __init__(self, split_dir: str, num_source_views: int,
                 resolution: tuple[int, int],
                 augmentations: dict | None = None):
        self.split_dir = Path(split_dir)
        self.num_source_views = num_source_views
        self.resolution = resolution  # (H, W)
        self.augmentations = augmentations or {}

        # Discover all sequence directories.
        self.sequences = sorted(
            d for d in self.split_dir.iterdir()
            if d.is_dir() and (d / "cameras.json").exists()
        )
        if not self.sequences:
            raise FileNotFoundError(f"No sequences found in {split_dir}")

        logger.info("Dataset %s: %d sequences", split_dir, len(self.sequences))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq_dir = self.sequences[idx]

        with open(seq_dir / "cameras.json", "r") as f:
            cam_data = json.load(f)

        views = cam_data["views"]
        h, w = self.resolution

        source_images = []
        source_intrinsics = []
        source_extrinsics = []
        target_image = None
        target_intrinsic = None
        target_extrinsic = None

        for filename, cam_info in views.items():
            img_path = seq_dir / "images" / filename
            img = self._load_image(img_path)

            intrinsic = torch.tensor(cam_info["intrinsic"], dtype=torch.float32)  # 3x3
            extrinsic = torch.tensor(cam_info["extrinsic"], dtype=torch.float32)  # 4x4

            if filename.startswith("target"):
                target_image = img
                target_intrinsic = intrinsic
                target_extrinsic = extrinsic
            elif filename.startswith("source"):
                source_images.append(img)
                source_intrinsics.append(intrinsic)
                source_extrinsics.append(extrinsic)

        if target_image is None:
            raise ValueError(f"No target image found in {seq_dir}")

        # Pad or truncate source views to exactly num_source_views.
        while len(source_images) < self.num_source_views:
            # Duplicate last source view if we have fewer than K.
            source_images.append(source_images[-1])
            source_intrinsics.append(source_intrinsics[-1])
            source_extrinsics.append(source_extrinsics[-1])
        source_images = source_images[:self.num_source_views]
        source_intrinsics = source_intrinsics[:self.num_source_views]
        source_extrinsics = source_extrinsics[:self.num_source_views]

        # Stack sources: [K, 3, H, W] and [K, 3, 3] / [K, 4, 4]
        source_images = torch.stack(source_images)
        source_intrinsics = torch.stack(source_intrinsics)
        source_extrinsics = torch.stack(source_extrinsics)

        # Apply augmentations.
        if self.augmentations.get("color_jitter", {}).get("enabled", False):
            source_images, target_image = self._apply_color_jitter(
                source_images, target_image, self.augmentations["color_jitter"])

        return {
            "source_images": source_images,           # [K, 3, H, W]
            "source_intrinsics": source_intrinsics,   # [K, 3, 3]
            "source_extrinsics": source_extrinsics,   # [K, 4, 4]
            "target_image": target_image,              # [3, H, W]
            "target_intrinsic": target_intrinsic,      # [3, 3]
            "target_extrinsic": target_extrinsic,      # [4, 4]
        }

    def _load_image(self, path: Path) -> torch.Tensor:
        """Load image, resize to target resolution, return as [3, H, W] float in [0, 1]."""
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        h, w = self.resolution
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0, 1].
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img

    @staticmethod
    def _apply_color_jitter(source_images: torch.Tensor, target_image: torch.Tensor,
                            jitter_cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply consistent random color jitter to all views in a sample."""
        from torchvision.transforms import functional as TF

        brightness = random.uniform(max(0, 1 - jitter_cfg.get("brightness", 0)),
                                    1 + jitter_cfg.get("brightness", 0))
        contrast = random.uniform(max(0, 1 - jitter_cfg.get("contrast", 0)),
                                  1 + jitter_cfg.get("contrast", 0))
        saturation = random.uniform(max(0, 1 - jitter_cfg.get("saturation", 0)),
                                    1 + jitter_cfg.get("saturation", 0))
        hue = random.uniform(-jitter_cfg.get("hue", 0), jitter_cfg.get("hue", 0))

        def apply(img: torch.Tensor) -> torch.Tensor:
            img = TF.adjust_brightness(img, brightness)
            img = TF.adjust_contrast(img, contrast)
            img = TF.adjust_saturation(img, saturation)
            img = TF.adjust_hue(img, hue)
            return torch.clamp(img, 0.0, 1.0)

        for k in range(source_images.shape[0]):
            source_images[k] = apply(source_images[k])
        target_image = apply(target_image)

        return source_images, target_image


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

def load_model(cfg: dict) -> nn.Module:
    """Load a pretrained feed-forward Gaussian splatting model.

    Supports DepthSplat and MVSplat.  Both are expected to expose a standard
    interface:

        model.forward(images, intrinsics, extrinsics, target_intrinsic, target_extrinsic)
            -> dict with "rendered_image" ([B, 3, H, W]) and optionally
               "gaussians" (position, scale, rotation, opacity, sh).

    If the installed package exposes a different interface, this wrapper adapts it.
    """
    model_type = cfg["model"]["type"]
    checkpoint_path = cfg["model"]["checkpoint"]
    num_views = cfg["model"]["num_source_views"]

    if model_type == "depthsplat":
        try:
            from depthsplat.model import DepthSplat

            model = DepthSplat.from_pretrained(checkpoint_path)
            logger.info("Loaded DepthSplat from %s", checkpoint_path)
        except ImportError:
            logger.error(
                "Cannot import depthsplat. Install it with: "
                "pip install git+https://github.com/cvg/depthsplat.git"
            )
            raise

    elif model_type == "mvsplat":
        try:
            from mvsplat.model import MVSplat

            model = MVSplat.from_pretrained(checkpoint_path)
            logger.info("Loaded MVSplat from %s", checkpoint_path)
        except ImportError:
            logger.error(
                "Cannot import mvsplat. Install it with: "
                "pip install git+https://github.com/donydchen/mvsplat.git"
            )
            raise

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Optionally freeze encoder.
    if cfg["model"].get("freeze_encoder", False):
        frozen_count = 0
        for name, param in model.named_parameters():
            # Convention: encoder parameters contain "encoder" in their name.
            if "encoder" in name.lower():
                param.requires_grad = False
                frozen_count += 1
        logger.info("Froze %d encoder parameters", frozen_count)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %d trainable / %d total (%.1f%%)",
                trainable, total, 100 * trainable / total if total > 0 else 0)

    return model


def forward_model(model: nn.Module, batch: dict[str, torch.Tensor],
                  model_type: str) -> dict[str, torch.Tensor]:
    """Run a forward pass through the model and return rendered target view.

    Both DepthSplat and MVSplat follow the general pattern:
        Input: source images + cameras + target camera
        Output: rendered image at the target viewpoint

    This function normalizes the interface for the training loop.
    """
    source_images = batch["source_images"]         # [B, K, 3, H, W]
    source_intrinsics = batch["source_intrinsics"] # [B, K, 3, 3]
    source_extrinsics = batch["source_extrinsics"] # [B, K, 4, 4]
    target_intrinsic = batch["target_intrinsic"]   # [B, 3, 3]
    target_extrinsic = batch["target_extrinsic"]   # [B, 4, 4]

    # Standard feed-forward GS model interface.
    output = model(
        images=source_images,
        intrinsics=source_intrinsics,
        extrinsics=source_extrinsics,
        target_intrinsic=target_intrinsic,
        target_extrinsic=target_extrinsic,
    )

    # Expect at minimum: output["rendered_image"] of shape [B, 3, H, W].
    if isinstance(output, dict):
        return output
    elif isinstance(output, torch.Tensor):
        return {"rendered_image": output}
    else:
        raise TypeError(f"Unexpected model output type: {type(output)}")


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
    """Compute 1 - SSIM as a loss.

    Uses a simple Gaussian window. pred and target are [B, 3, H, W] in [0, 1].
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian kernel.
    coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
    coords -= window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)  # [ws, ws]
    window = window.unsqueeze(0).unsqueeze(0)  # [1, 1, ws, ws]
    window = window.expand(3, 1, -1, -1)      # [3, 1, ws, ws]

    pad = window_size // 2

    mu_pred = F.conv2d(pred, window, padding=pad, groups=3)
    mu_target = F.conv2d(target, window, padding=pad, groups=3)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=3) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=3) - mu_target_sq
    sigma_cross = F.conv2d(pred * target, window, padding=pad, groups=3) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """Combined photometric loss: L1 + SSIM + optional LPIPS."""

    def __init__(self, l1_weight: float = 0.8, ssim_weight: float = 0.2,
                 lpips_weight: float = 0.0, lpips_net: str = "vgg",
                 opacity_reg_weight: float = 0.0, scale_reg_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.opacity_reg_weight = opacity_reg_weight
        self.scale_reg_weight = scale_reg_weight

        self.lpips_fn = None
        if lpips_weight > 0:
            import lpips as lpips_module
            self.lpips_fn = lpips_module.LPIPS(net=lpips_net)
            # Freeze LPIPS — it is used only as a metric/loss, not trained.
            for p in self.lpips_fn.parameters():
                p.requires_grad = False

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                model_output: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred:   Rendered image [B, 3, H, W] in [0, 1].
            target: Ground-truth image [B, 3, H, W] in [0, 1].
            model_output: Optional dict containing "opacities" and "scales"
                          tensors for regularization.

        Returns:
            Dict with "total" and per-component losses.
        """
        losses = {}

        # L1 loss.
        l1 = F.l1_loss(pred, target)
        losses["l1"] = l1

        # SSIM loss.
        ssim = ssim_loss(pred, target)
        losses["ssim"] = ssim

        total = self.l1_weight * l1 + self.ssim_weight * ssim

        # LPIPS perceptual loss.
        if self.lpips_fn is not None and self.lpips_weight > 0:
            # LPIPS expects images in [-1, 1].
            lpips_val = self.lpips_fn(pred * 2 - 1, target * 2 - 1).mean()
            losses["lpips"] = lpips_val
            total = total + self.lpips_weight * lpips_val

        # Opacity regularization: encourage opacity away from 0.5 (either opaque or transparent).
        if self.opacity_reg_weight > 0 and model_output and "opacities" in model_output:
            opacities = model_output["opacities"]
            # Binary entropy-like penalty: minimal at 0 and 1, maximal at 0.5.
            opacity_reg = (-opacities * torch.log(opacities + 1e-6)
                           - (1 - opacities) * torch.log(1 - opacities + 1e-6)).mean()
            losses["opacity_reg"] = opacity_reg
            total = total + self.opacity_reg_weight * opacity_reg

        # Scale regularization: penalize excessively large Gaussians.
        if self.scale_reg_weight > 0 and model_output and "scales" in model_output:
            scales = model_output["scales"]
            scale_reg = torch.mean(torch.relu(scales - 0.1))
            losses["scale_reg"] = scale_reg
            total = total + self.scale_reg_weight * scale_reg

        losses["total"] = total
        return losses


# ---------------------------------------------------------------------------
# Learning rate scheduler with warmup
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing to min_lr."""

    def __init__(self, optimizer: torch.optim.Optimizer,
                 warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.step_count <= self.warmup_steps:
                # Linear warmup.
                lr = base_lr * self.step_count / max(1, self.warmup_steps)
            else:
                # Cosine decay.
                progress = (self.step_count - self.warmup_steps) / \
                           max(1, self.total_steps - self.warmup_steps)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Logger backends
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Thin wrapper around TensorBoard / W&B for metric logging."""

    def __init__(self, backend: str, log_dir: str, project: str | None = None,
                 entity: str | None = None, config: dict | None = None):
        self.backend = backend
        self.writer = None

        if backend == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            logger.info("TensorBoard logging to %s", log_dir)
        elif backend == "wandb":
            import wandb
            wandb.init(project=project or "heimdall-finetune",
                       entity=entity, dir=log_dir, config=config)
            self.writer = wandb
            logger.info("W&B logging (project=%s)", project)
        else:
            logger.warning("Unknown logger backend: %s", backend)

    def log_scalar(self, tag: str, value: float, step: int):
        if self.backend == "tensorboard" and self.writer:
            self.writer.add_scalar(tag, value, step)
        elif self.backend == "wandb" and self.writer:
            self.writer.log({tag: value}, step=step)

    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """Log an image tensor [3, H, W] or [H, W, 3] in [0, 1]."""
        if self.backend == "tensorboard" and self.writer:
            if image.ndim == 3 and image.shape[0] == 3:
                pass  # Already CHW
            elif image.ndim == 3 and image.shape[2] == 3:
                image = image.permute(2, 0, 1)
            self.writer.add_image(tag, image, step)
        elif self.backend == "wandb" and self.writer:
            import wandb
            if image.ndim == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            self.writer.log({tag: wandb.Image(image.cpu().numpy())}, step=step)

    def close(self):
        if self.backend == "tensorboard" and self.writer:
            self.writer.close()
        elif self.backend == "wandb" and self.writer:
            self.writer.finish()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: WarmupCosineScheduler, epoch: int,
                    best_val_loss: float, path: str):
    """Save a training checkpoint."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_step_count": scheduler.step_count,
        "best_val_loss": best_val_loss,
    }, path)
    logger.info("Saved checkpoint: %s", path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    scheduler: WarmupCosineScheduler | None = None) -> dict:
    """Load a training checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and "scheduler_step_count" in ckpt:
        scheduler.step_count = ckpt["scheduler_step_count"]
    logger.info("Loaded checkpoint: %s (epoch %d)", path, ckpt.get("epoch", -1))
    return ckpt


def prune_checkpoints(checkpoint_dir: Path, keep_last_n: int, best_path: str | None = None):
    """Delete old checkpoints, keeping the N most recent plus the best."""
    ckpts = sorted(checkpoint_dir.glob("epoch_*.pth"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for ckpt in ckpts[keep_last_n:]:
        if best_path and str(ckpt) == best_path:
            continue
        ckpt.unlink()
        logger.debug("Pruned checkpoint: %s", ckpt)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, loss_fn: CombinedLoss,
             model_type: str, device: torch.device) -> dict[str, float]:
    """Run validation and return average losses."""
    model.eval()
    totals: dict[str, float] = {}
    count = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        output = forward_model(model, batch, model_type)
        rendered = output["rendered_image"]
        target = batch["target_image"]

        losses = loss_fn(rendered, target, output)
        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        count += 1

    model.train()
    return {k: v / max(1, count) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(cfg: dict):
    """Main fine-tuning entry point."""
    # Seed everything.
    seed = cfg["training"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Paths.
    dataset_dir = Path(cfg["data"]["dataset_dir"])
    checkpoint_dir = Path(cfg["output"]["checkpoint_dir"])
    log_dir = Path(cfg["output"]["log_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    resolution = tuple(cfg["model"]["resolution"])
    num_source_views = cfg["data"]["num_source_views"]

    # Datasets.
    train_ds = GaussianFinetuneDataset(
        split_dir=str(dataset_dir / "train"),
        num_source_views=num_source_views,
        resolution=resolution,
        augmentations=cfg["data"].get("augmentations"),
    )
    val_ds = GaussianFinetuneDataset(
        split_dir=str(dataset_dir / "val"),
        num_source_views=num_source_views,
        resolution=resolution,
        augmentations=None,  # No augmentation for validation.
    )

    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    # Model.
    model = load_model(cfg)
    model = model.to(device)
    model.train()

    # Optimizer.
    lr = float(cfg["training"]["lr"])
    weight_decay = float(cfg["training"].get("weight_decay", 0.01))
    betas = tuple(cfg["training"].get("betas", [0.9, 0.999]))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    # Scheduler.
    epochs = cfg["training"]["epochs"]
    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    steps_per_epoch = len(train_loader) // grad_accum
    total_steps = steps_per_epoch * epochs
    warmup_steps = cfg["training"].get("warmup_steps", 500)
    min_lr = float(cfg["training"].get("min_lr", 1e-6))

    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr)

    # Loss.
    loss_cfg = cfg["loss"]
    loss_fn = CombinedLoss(
        l1_weight=loss_cfg.get("l1_weight", 0.8),
        ssim_weight=loss_cfg.get("ssim_weight", 0.2),
        lpips_weight=loss_cfg.get("lpips_weight", 0.0),
        lpips_net=loss_cfg.get("lpips_net", "vgg"),
        opacity_reg_weight=loss_cfg.get("opacity_reg_weight", 0.0),
        scale_reg_weight=loss_cfg.get("scale_reg_weight", 0.0),
    )
    loss_fn = loss_fn.to(device)

    # Mixed precision.
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Gradient clipping.
    grad_clip = cfg["training"].get("grad_clip", 0.0)

    # Logger.
    train_logger = TrainingLogger(
        backend=cfg["output"].get("logger", "tensorboard"),
        log_dir=str(log_dir),
        project=cfg["output"].get("wandb_project"),
        entity=cfg["output"].get("wandb_entity"),
        config=cfg,
    )

    # Resume from checkpoint if exists.
    best_val_loss = float("inf")
    start_epoch = 0
    latest_ckpt = checkpoint_dir / "latest.pth"
    if latest_ckpt.exists():
        ckpt_data = load_checkpoint(str(latest_ckpt), model, optimizer, scheduler)
        start_epoch = ckpt_data.get("epoch", 0) + 1
        best_val_loss = ckpt_data.get("best_val_loss", float("inf"))
        logger.info("Resuming from epoch %d", start_epoch)

    # Save config alongside checkpoints.
    with open(checkpoint_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    model_type = cfg["model"]["type"]
    save_interval = cfg["output"].get("save_interval", 10)
    eval_interval = cfg["output"].get("eval_interval", 5)
    keep_last_n = cfg["output"].get("keep_last_n", 5)

    global_step = start_epoch * steps_per_epoch

    logger.info("Starting training: %d epochs, %d steps/epoch, %d total steps",
                epochs, steps_per_epoch, total_steps)

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        epoch_losses: dict[str, float] = {}
        batch_count = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = forward_model(model, batch, model_type)
                rendered = output["rendered_image"]
                target = batch["target_image"]

                losses = loss_fn(rendered, target, output)
                loss = losses["total"] / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            # Accumulate epoch stats.
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            batch_count += 1

            # Log per-step metrics.
            if global_step % 50 == 0:
                train_logger.log_scalar("train/loss", losses["total"].item(), global_step)
                train_logger.log_scalar("train/lr", scheduler.get_lr(), global_step)
                for k, v in losses.items():
                    if k != "total":
                        train_logger.log_scalar(f"train/{k}", v.item(), global_step)

        # End of epoch.
        epoch_time = time.time() - epoch_start
        avg_losses = {k: v / max(1, batch_count) for k, v in epoch_losses.items()}

        logger.info("Epoch %d/%d  loss=%.4f  lr=%.2e  time=%.1fs",
                     epoch + 1, epochs, avg_losses.get("total", 0),
                     scheduler.get_lr(), epoch_time)

        # Validation.
        if (epoch + 1) % eval_interval == 0 or epoch == epochs - 1:
            val_losses = validate(model, val_loader, loss_fn, model_type, device)
            logger.info("  Val loss=%.4f (l1=%.4f, ssim=%.4f)",
                        val_losses.get("total", 0), val_losses.get("l1", 0),
                        val_losses.get("ssim", 0))
            for k, v in val_losses.items():
                train_logger.log_scalar(f"val/{k}", v, global_step)

            # Log sample images.
            if val_loader.dataset and len(val_loader.dataset) > 0:
                sample = val_loader.dataset[0]
                sample_batch = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                                for k, v in sample.items()}
                with torch.no_grad():
                    sample_out = forward_model(model, sample_batch, model_type)
                train_logger.log_image("val/rendered", sample_out["rendered_image"][0].cpu(), global_step)
                train_logger.log_image("val/target", sample["target_image"], global_step)

            # Track best model.
            val_total = val_losses.get("total", float("inf"))
            if val_total < best_val_loss:
                best_val_loss = val_total
                best_path = str(checkpoint_dir / "best.pth")
                save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, best_path)

        # Save periodic checkpoint.
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            ckpt_path = str(checkpoint_dir / f"epoch_{epoch + 1:04d}.pth")
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, ckpt_path)
            # Also save as latest for resume.
            save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, str(latest_ckpt))
            if keep_last_n > 0:
                prune_checkpoints(checkpoint_dir, keep_last_n,
                                  best_path=str(checkpoint_dir / "best.pth"))

    train_logger.close()
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Checkpoints: %s", checkpoint_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune a feed-forward Gaussian splatting model on Heimdall rig data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to finetune_config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (overrides auto-resume)")
    parser.add_argument("--verbose", action="store_true")

    args, unknown = parser.parse_known_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config with CLI overrides (e.g. --training.lr=5e-5 passed as extra args).
    overrides = []
    for u in unknown:
        if u.startswith("--") and "=" in u:
            overrides.append(u.lstrip("-"))
        elif u.startswith("--") and not u.startswith("---"):
            # Handle "--key value" form by combining with next arg.
            pass

    cfg = load_config(args.config, overrides)

    if args.resume:
        # Copy resume checkpoint as latest so the training loop picks it up.
        import shutil
        ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.resume, str(ckpt_dir / "latest.pth"))

    train(cfg)


if __name__ == "__main__":
    main()
