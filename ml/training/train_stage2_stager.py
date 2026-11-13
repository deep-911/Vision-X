"""
VisionX: Stage 2 Specialist Severity Stager Training Pipeline (PyTorch)
========================================================================
Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
Team: Indian_Gryffindor

Directly mirrors the team's MATLAB implementation:
  matlab/training/train_stage2_stager.m

Trains two decoupled specialist classifiers:
  - Part A (Referable Specialist Stager - Stage 2B):
      * Moderate NPDR (Grade 2)
      * Severe NPDR (Grade 3)
      * Proliferative DR (Grade 4)
  - Part B (Non-Referable Specialist Stager - Stage 2A):
      * No DR (Grade 0)
      * Mild NPDR (Grade 1)

Decoupling triage from grading allows each specialist model to focus solely
on the nuanced morphological boundaries between adjacent severity stages.
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
from typing import Any, Dict, List, Tuple

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
    STAGE2_REF_CLASSES,
    STAGE2_NONREF_CLASSES,
    Stage2SeverityStager,
    build_stage2_stager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("visionx.train_stage2_stager")


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SubCohortDataset(Dataset):
    def __init__(self, file_paths: List[Path], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
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


def discover_subcohort(data_dir: Path, target_classes: List[str]) -> Tuple[List[Path], List[int]]:
    files: List[Path] = []
    labels: List[int] = []

    for folder in data_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name
        matched_idx = None
        for i, target_name in enumerate(target_classes):
            if folder_name.lower() == target_name.lower():
                matched_idx = i
                break

        if matched_idx is None:
            continue

        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img_file in folder.glob(ext):
                files.append(img_file)
                labels.append(matched_idx)

    return files, labels


def train_stage2(
    data_dir: Path,
    output_model_path: Path,
    report_path: Path,
    backbone: str = "resnet18",
    epochs_ref: int = 4,
    epochs_non: int = 3,
    batch_size: int = 24,
    lr: float = 1e-4,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    seed_everything(42)
    logger.info("===================================================================")
    logger.info(" VisionX: Stage 2 Specialist Severity Stager Training (PyTorch)")
    logger.info(" Mirroring: matlab/training/train_stage2_stager.m")
    logger.info(" Backbone : %s | Device: %s", backbone, device)
    logger.info("===================================================================")

    # Transform pipelines
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = build_stage2_stager(backbone=backbone, pretrained=True).to(device)

    # ---------------------------------------------------------------------
    # PART A: REFERABLE SPECIALIST STAGER (MODERATE vs SEVERE vs PDR)
    # ---------------------------------------------------------------------
    logger.info("\n[PART A] Training Referable Stager (Moderate vs Severe vs PDR)...")
    ref_files, ref_labels = discover_subcohort(data_dir, STAGE2_REF_CLASSES)

    ref_acc = 0.0
    cm_ref = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    if len(ref_files) > 0:
        indices = list(range(len(ref_files)))
        random.shuffle(indices)
        split_pt = int(len(indices) * 0.80)
        train_idx, val_idx = indices[:split_pt], indices[split_pt:]

        # Oversample Severe (+2x) and PDR (+1x) to counter class imbalance (mirrors MATLAB)
        train_sev = [i for i in train_idx if ref_labels[i] == 1]  # Severe
        train_pdr = [i for i in train_idx if ref_labels[i] == 2]  # PDR
        train_idx = train_idx + train_sev * 2 + train_pdr

        train_ds_ref = SubCohortDataset([ref_files[i] for i in train_idx], [ref_labels[i] for i in train_idx], train_transform)
        val_ds_ref = SubCohortDataset([ref_files[i] for i in val_idx], [ref_labels[i] for i in val_idx], val_transform)

        loader_train_ref = DataLoader(train_ds_ref, batch_size=batch_size, shuffle=True)
        loader_val_ref = DataLoader(val_ds_ref, batch_size=batch_size, shuffle=False)

        opt_ref = torch.optim.Adam(
            list(model.base.parameters()) + list(model.head_referable.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        crit_ref = nn.CrossEntropyLoss()

        for epoch in range(1, epochs_ref + 1):
            model.train()
            running_loss = 0.0
            corr = 0
            tot = 0
            for imgs, targets in loader_train_ref:
                imgs, targets = imgs.to(device), targets.to(device)
                opt_ref.zero_grad()
                out = model.forward_referable(imgs)
                loss = crit_ref(out, targets)
                loss.backward()
                opt_ref.step()

                running_loss += loss.item() * imgs.size(0)
                corr += (out.argmax(dim=-1) == targets).sum().item()
                tot += targets.size(0)

            logger.info("  Referable Epoch [%d/%d] - Loss: %.4f - Train Acc: %.2f%%",
                        epoch, epochs_ref, running_loss / max(tot, 1), (corr / max(tot, 1)) * 100.0)

        # Evaluate Referable Stager
        model.eval()
        preds_ref, trues_ref = [], []
        with torch.no_grad():
            for imgs, targets in loader_val_ref:
                imgs = imgs.to(device)
                out = model.forward_referable(imgs)
                preds_ref.extend(out.argmax(dim=-1).cpu().tolist())
                trues_ref.extend(targets.tolist())

        cm_ref = [[0 for _ in range(3)] for _ in range(3)]
        for t, p in zip(trues_ref, preds_ref):
            cm_ref[t][p] += 1
        ref_acc = sum(cm_ref[i][i] for i in range(3)) / max(len(trues_ref), 1)
        logger.info("  --> Referable Stager Validation Accuracy: %.2f%%", ref_acc * 100.0)

    # ---------------------------------------------------------------------
    # PART B: NON-REFERABLE SPECIALIST STAGER (NO DR vs MILD)
    # ---------------------------------------------------------------------
    logger.info("\n[PART B] Training Non-Referable Stager (No_DR vs Mild)...")
    non_files, non_labels = discover_subcohort(data_dir, STAGE2_NONREF_CLASSES)

    non_acc = 0.0
    cm_non = [[0, 0], [0, 0]]

    if len(non_files) > 0:
        indices = list(range(len(non_files)))
        random.shuffle(indices)
        split_pt = int(len(indices) * 0.80)
        train_idx, val_idx = indices[:split_pt], indices[split_pt:]

        # Oversample Mild (+3x) (mirrors MATLAB)
        train_mild = [i for i in train_idx if non_labels[i] == 1]
        train_idx = train_idx + train_mild * 3

        train_ds_non = SubCohortDataset([non_files[i] for i in train_idx], [non_labels[i] for i in train_idx], train_transform)
        val_ds_non = SubCohortDataset([non_files[i] for i in val_idx], [non_labels[i] for i in val_idx], val_transform)

        loader_train_non = DataLoader(train_ds_non, batch_size=batch_size, shuffle=True)
        loader_val_non = DataLoader(val_ds_non, batch_size=batch_size, shuffle=False)

        opt_non = torch.optim.Adam(
            list(model.base.parameters()) + list(model.head_non_referable.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        crit_non = nn.CrossEntropyLoss()

        for epoch in range(1, epochs_non + 1):
            model.train()
            running_loss = 0.0
            corr = 0
            tot = 0
            for imgs, targets in loader_train_non:
                imgs, targets = imgs.to(device), targets.to(device)
                opt_non.zero_grad()
                out = model.forward_non_referable(imgs)
                loss = crit_non(out, targets)
                loss.backward()
                opt_non.step()

                running_loss += loss.item() * imgs.size(0)
                corr += (out.argmax(dim=-1) == targets).sum().item()
                tot += targets.size(0)

            logger.info("  Non-Referable Epoch [%d/%d] - Loss: %.4f - Train Acc: %.2f%%",
                        epoch, epochs_non, running_loss / max(tot, 1), (corr / max(tot, 1)) * 100.0)

        # Evaluate Non-Referable Stager
        model.eval()
        preds_non, trues_non = [], []
        with torch.no_grad():
            for imgs, targets in loader_val_non:
                imgs = imgs.to(device)
                out = model.forward_non_referable(imgs)
                preds_non.extend(out.argmax(dim=-1).cpu().tolist())
                trues_non.extend(targets.tolist())

        cm_non = [[0 for _ in range(2)] for _ in range(2)]
        for t, p in zip(trues_non, preds_non):
            cm_non[t][p] += 1
        non_acc = sum(cm_non[i][i] for i in range(2)) / max(len(trues_non), 1)
        logger.info("  --> Non-Referable Stager Validation Accuracy: %.2f%%", non_acc * 100.0)

    # Save Stage 2 Checkpoint
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": backbone,
            "classes_referable": STAGE2_REF_CLASSES,
            "classes_non_referable": STAGE2_NONREF_CLASSES,
            "acc_referable": round(ref_acc, 4),
            "acc_non_referable": round(non_acc, 4),
        },
        str(output_model_path),
    )
    logger.info("\n[SAVE] Stage 2 Specialist Stagers saved to: %s", output_model_path)

    # Save JSON Report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "model_type": f"Stage 2 Specialist Severity Stagers ({backbone})",
        "referable_stager": {
            "classes": STAGE2_REF_CLASSES,
            "accuracy": round(ref_acc, 4),
            "confusion_matrix": cm_ref,
        },
        "non_referable_stager": {
            "classes": STAGE2_NONREF_CLASSES,
            "accuracy": round(non_acc, 4),
            "confusion_matrix": cm_non,
        },
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    logger.info("[SAVE] Stage 2 evaluation report saved to: %s", report_path)

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stage 2 Severity Stager")
    parser.add_argument("--data_dir", type=str, default=str(PROJECT_ROOT / "data" / "processed" / "aptos_true_512"))
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--epochs_ref", type=int, default=4)
    parser.add_argument("--epochs_non", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    out_pt = PROJECT_ROOT / "models" / "production" / "visionx_stage2_stager.pt"
    out_rep = PROJECT_ROOT / "artifacts" / "reports" / "stage2_stager_evaluation.json"

    train_stage2(
        data_dir=Path(args.data_dir),
        output_model_path=out_pt,
        report_path=out_rep,
        backbone=args.backbone,
        epochs_ref=args.epochs_ref,
        epochs_non=args.epochs_non,
        batch_size=args.batch_size,
        lr=args.lr,
    )
