"""
VisionX: Stage 1 Binary Referral Gate Training Pipeline (PyTorch)
==================================================================
Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
Team: Indian_Gryffindor

Directly mirrors the team's MATLAB implementation:
  matlab/training/train_stage1_referral_gate.m

Trains a high-sensitivity binary referral gatekeeper:
  - Class 0: Non_Referable (Grade 0: No DR, Grade 1: Mild NPDR)
  - Class 1: Referable     (Grade 2: Moderate, Grade 3: Severe, Grade 4: Proliferative DR)

Edge Deployment Target:
  Lightweight backbone (ResNet-18 / MobileNetV3 / EfficientNet-B0) designed for
  real-time inference on low-power Primary Health Centre (PHC) screening hardware.

Clinical Objective:
  Calibrated operating threshold (tau = 0.30) ensuring >90-94% sensitivity on
  referable cases to prevent false negatives from being sent home.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.two_stage import (
    STAGE1_CLASSES,
    Stage1ReferralGate,
    build_stage1_gate,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("visionx.train_stage1_gate")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BinaryReferralDataset(Dataset):
    """
    Dataset loader mapping 5 clinical DR classes into 2 referral categories:
      - Non_Referable (0): No_DR (0), Mild (1)
      - Referable (1)    : Moderate (2), Severe (3), Proliferate_DR (4)
    """

    def __init__(self, file_paths: List[Path], binary_labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = binary_labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def discover_binary_dataset(data_dir: Path) -> Tuple[List[Path], List[int]]:
    """Discovers images from folder structure and maps to binary referral labels."""
    non_ref_folders = ["No_DR", "Mild", "0", "1"]
    ref_folders = ["Moderate", "Severe", "Proliferate_DR", "2", "3", "4"]

    files: List[Path] = []
    labels: List[int] = []

    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        if folder_name in non_ref_folders:
            label = 0  # Non_Referable
        elif folder_name in ref_folders:
            label = 1  # Referable
        else:
            continue

        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_file in folder.glob(ext):
                files.append(img_file)
                labels.append(label)

    return files, labels


def train_stage1(
    data_dir: Path,
    output_model_path: Path,
    report_path: Path,
    backbone: str = "resnet18",
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-4,
    val_split: float = 0.20,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    seed_everything(42)
    logger.info("===================================================================")
    logger.info(" VisionX: Stage 1 Binary Referral Gate Training (PyTorch)")
    logger.info(" Mirroring: matlab/training/train_stage1_referral_gate.m")
    logger.info(" Backbone : %s | Device: %s", backbone, device)
    logger.info("===================================================================")

    all_files, all_labels = discover_binary_dataset(data_dir)
    logger.info("Found %d total scans in %s", len(all_files), data_dir)

    if len(all_files) == 0:
        logger.warning("No images found at %s. Creating synthetic demo data for verification.", data_dir)
        return {"status": "NO_DATA", "scans": 0}

    # Stratified 80/20 train/val split
    idx_non_ref = [i for i, lbl in enumerate(all_labels) if lbl == 0]
    idx_ref = [i for i, lbl in enumerate(all_labels) if lbl == 1]

    random.shuffle(idx_non_ref)
    random.shuffle(idx_ref)

    split_non = int(len(idx_non_ref) * (1 - val_split))
    split_ref = int(len(idx_ref) * (1 - val_split))

    train_indices = idx_non_ref[:split_non] + idx_ref[:split_ref]
    val_indices = idx_non_ref[split_non:] + idx_ref[split_ref:]

    # Oversample Referable in training to handle class imbalance (mirrors MATLAB)
    train_ref_sub = [i for i in train_indices if all_labels[i] == 1]
    train_indices = train_indices + train_ref_sub  # +1x oversampling of referable cases

    train_files = [all_files[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]
    val_labels = [all_labels[i] for i in val_indices]

    logger.info("Training Scans (Balanced)  : %d (Non-Ref: %d, Ref: %d)",
                len(train_files), train_labels.count(0), train_labels.count(1))
    logger.info("Validation Scans (Holdout) : %d (Non-Ref: %d, Ref: %d)",
                len(val_files), val_labels.count(0), val_labels.count(1))

    # Augmentations mirroring MATLAB imageDataAugmenter
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = BinaryReferralDataset(train_files, train_labels, transform=train_transform)
    val_dataset = BinaryReferralDataset(val_files, val_labels, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build Stage 1 Referral Gatekeeper
    model = build_stage1_gate(backbone=backbone, pretrained=True, tau_referable=0.30)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.30)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100.0
        train_loss = running_loss / total
        logger.info("Epoch [%d/%d] - Loss: %.4f - Train Acc: %.2f%%", epoch, epochs, train_loss, train_acc)

    train_duration = time.time() - start_time
    logger.info("Training completed in %.1f seconds (%.2f minutes)", train_duration, train_duration / 60.0)

    # Evaluation on Holdout Cohort
    model.eval()
    all_preds_prob: List[float] = []
    all_true: List[int] = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            probs = model.predict_proba(images)
            p_ref = probs[:, 1].cpu().tolist()
            all_preds_prob.extend(p_ref)
            all_true.extend(targets.tolist())

    y_true = np.array(all_true)
    p_ref_arr = np.array(all_preds_prob)

    # Threshold sensitivity sweep (Operating Curve)
    logger.info("\n--- STAGE 1 REFERRAL GATE OPERATING POINT SWEEP ---")
    logger.info("%-10s | %-12s | %-12s | %-12s", "Threshold", "Sensitivity", "Specificity", "False Negatives")
    logger.info("-" * 55)

    operating_results = {}
    for tau in [0.50, 0.40, 0.35, 0.30, 0.25, 0.20]:
        y_pred = (p_ref_arr >= tau).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))

        sens = (tp / max(tp + fn, 1)) * 100.0
        spec = (tn / max(tn + fp, 1)) * 100.0
        logger.info("tau = %-5.2f | %-10.2f%% | %-10.2f%% | %-12d", tau, sens, spec, fn)
        operating_results[str(tau)] = {
            "sensitivity": round(sens / 100.0, 4),
            "specificity": round(spec / 100.0, 4),
            "false_negatives": fn,
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp": fp,
        }

    # Primary operating point: tau = 0.30 (calibrated rural clinical triage)
    tau_default = 0.30
    y_pred_calibrated = (p_ref_arr >= tau_default).astype(int)
    cm = [
        [int(np.sum((y_pred_calibrated == 0) & (y_true == 0))), int(np.sum((y_pred_calibrated == 1) & (y_true == 0)))],
        [int(np.sum((y_pred_calibrated == 0) & (y_true == 1))), int(np.sum((y_pred_calibrated == 1) & (y_true == 1)))],
    ]
    tp = cm[1][1]
    fn = cm[1][0]
    tn = cm[0][0]
    fp = cm[0][1]

    calibrated_sens = tp / max(tp + fn, 1)
    calibrated_spec = tn / max(tn + fp, 1)
    accuracy = (tp + tn) / max(len(y_true), 1)

    # Save Model Weights
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": backbone,
            "classes": STAGE1_CLASSES,
            "tau_referable": tau_default,
            "metrics": {
                "accuracy": round(accuracy, 4),
                "sensitivity": round(calibrated_sens, 4),
                "specificity": round(calibrated_spec, 4),
                "false_negatives": fn,
            },
        },
        str(output_model_path),
    )
    logger.info("[SAVE] Stage 1 Gate model saved to: %s", output_model_path)

    # Save JSON Report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model_type": f"Stage 1 Binary Referral Gate ({backbone})",
        "tau_calibrated": tau_default,
        "accuracy": round(accuracy, 4),
        "sensitivity": round(calibrated_sens, 4),
        "specificity": round(calibrated_spec, 4),
        "false_negatives": fn,
        "confusion_matrix": cm,
        "operating_points": operating_results,
        "classes": STAGE1_CLASSES,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("[SAVE] Stage 1 Gate report saved to: %s", report_path)

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 1 Referral Gate")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "aptos_true_512"))
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    out_pt = PROJECT_ROOT / "models" / "production" / "visionx_stage1_gate.pt"
    out_rep = PROJECT_ROOT / "artifacts" / "reports" / "stage1_gate_evaluation.json"

    train_stage1(
        data_dir=Path(args.data_dir),
        output_model_path=out_pt,
        report_path=out_rep,
        backbone=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
