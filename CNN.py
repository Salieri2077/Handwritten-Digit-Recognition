#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Handwritten Digit Recognition (MNIST) with a paper-inspired "DenseNet-BC" CNN.

Architecture reference:
  - "Densely Connected Convolutional Networks" (Huang et al., CVPR 2017)
    DenseNet-BC: bottleneck (1x1 conv) + compression in transition layers.

This script is intentionally self-contained in ONE .py file:
  - Defines the model (DenseNet-BC adapted for MNIST)
  - Trains + evaluates + saves best checkpoint
  - Optional single-image prediction

Run:
  python densenet_mnist.py --epochs 12 --batch-size 128
Predict:
  python densenet_mnist.py --predict path/to/image.png

Notes:
  - MNIST downloads automatically via torchvision.
  - For best results, keep the input as a centered digit on a black background.
"""

import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m{s:02d}s"


# ----------------------------
# DenseNet-BC (paper-inspired)
# ----------------------------
class DenseLayer(nn.Module):
    """
    DenseNet-BC layer: BN -> ReLU -> 1x1 Conv (bottleneck) ->
                       BN -> ReLU -> 3x3 Conv -> (Dropout) -> concat
    """
    def __init__(self, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        # DenseNet concatenation
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int, drop_rate: float):
        super().__init__()
        layers = []
        c = in_channels
        for _ in range(num_layers):
            layers.append(DenseLayer(c, growth_rate, bn_size, drop_rate))
            c += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Transition(nn.Module):
    """
    Transition layer with compression theta:
      BN -> ReLU -> 1x1 Conv (reduce channels) -> AvgPool(2)
    """
    def __init__(self, in_channels: int, theta: float):
        super().__init__()
        out_channels = int(in_channels * theta)
        out_channels = max(out_channels, 1)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(F.relu(self.bn(x), inplace=True))
        x = self.pool(x)
        return x


class DenseNetBC_MNIST(nn.Module):
    """
    A DenseNet-BC variant adapted for MNIST (1x28x28).
    - Uses two transition downsamples: 28->14->7, then global average pool.
    """
    def __init__(
        self,
        growth_rate: int = 16,
        block_config: Tuple[int, int, int] = (8, 12, 16),
        init_channels: int = 32,
        bn_size: int = 4,
        theta: float = 0.5,
        drop_rate: float = 0.1,
        num_classes: int = 10,
    ):
        super().__init__()

        # "Stem" (lightweight for MNIST)
        self.stem = nn.Sequential(
            nn.Conv2d(1, init_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )

        c = init_channels

        # Dense Block 1
        self.db1 = DenseBlock(block_config[0], c, growth_rate, bn_size, drop_rate)
        c = self.db1.out_channels
        self.tr1 = Transition(c, theta)
        c = self.tr1.out_channels

        # Dense Block 2
        self.db2 = DenseBlock(block_config[1], c, growth_rate, bn_size, drop_rate)
        c = self.db2.out_channels
        self.tr2 = Transition(c, theta)
        c = self.tr2.out_channels

        # Dense Block 3 (no transition after)
        self.db3 = DenseBlock(block_config[2], c, growth_rate, bn_size, drop_rate)
        c = self.db3.out_channels

        self.final_bn = nn.BatchNorm2d(c)
        self.classifier = nn.Linear(c, num_classes)

        # Init following common DenseNet practice
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.db1(x)
        x = self.tr1(x)
        x = self.db2(x)
        x = self.tr2(x)
        x = self.db3(x)
        x = F.relu(self.final_bn(x), inplace=True)
        x = F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
        return self.classifier(x)


# ----------------------------
# Training/Eval
# ----------------------------
@dataclass
class TrainConfig:
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    label_smoothing: float
    num_workers: int
    seed: int
    amp: bool
    out_dir: str
    save_name: str
    device: str


def get_dataloaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    # MNIST mean/std are commonly used; keep consistent normalization.
    mean, std = 0.1307, 0.3081

    train_tf = transforms.Compose([
        transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=8),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
        # Small random erasing can improve robustness, but keep conservative.
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.0), value=0.0),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=train_tf)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bsz = x.size(0)
        loss_meter += loss.item() * bsz
        acc_meter += accuracy(logits.detach(), y) * bsz
        n += bsz

    return loss_meter / n, acc_meter / n


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_meter = 0.0
    acc_meter = 0.0
    n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bsz = x.size(0)
        loss_meter += loss.item() * bsz
        acc_meter += accuracy(logits, y) * bsz
        n += bsz

    return loss_meter / n, acc_meter / n


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_acc: float) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_acc": best_acc,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


# ----------------------------
# Prediction (single image)
# ----------------------------
@torch.no_grad()
def predict_one_image(model: nn.Module, image_path: str, device: torch.device) -> Tuple[int, torch.Tensor]:
    from PIL import Image

    mean, std = 0.1307, 0.3081
    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    pred = int(probs.argmax().item())
    return pred, probs


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DenseNet-BC MNIST (single-file)")

    # Training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP)")

    # Model (DenseNet-BC knobs)
    p.add_argument("--growth-rate", type=int, default=16)
    p.add_argument("--init-channels", type=int, default=32)
    p.add_argument("--bn-size", type=int, default=4)
    p.add_argument("--theta", type=float, default=0.5, help="compression factor in transition layers")
    p.add_argument("--drop-rate", type=float, default=0.1)
    p.add_argument("--blocks", type=str, default="8,12,16", help="comma-separated layers per dense block, e.g. 6,12,24")

    # IO
    p.add_argument("--out-dir", type=str, default="./checkpoints")
    p.add_argument("--save-name", type=str, default="densenet_bc_mnist.pt")

    # Predict
    p.add_argument("--predict", type=str, default="", help="path to a single image to predict")
    p.add_argument("--checkpoint", type=str, default="", help="path to a checkpoint to load for prediction/eval")

    return p.parse_args()

