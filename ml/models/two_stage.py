"""
VisionX: Authentic Two-Stage Retinal Screening Pipeline (PyTorch)
=================================================================
Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
Team: Indian_Gryffindor

Implements the authentic Two-Stage Pipeline designed for rural Primary Health Centre (PHC)
telemedicine screening, directly mirroring the team's MATLAB scripts:
  - matlab/training/train_stage1_referral_gate.m
  - matlab/training/train_stage2_stager.m
  - matlab/evaluation/evaluate_two_stage.m

Two-Stage Clinical Architecture:
--------------------------------
1. Stage 1: Referral Gatekeeper (Binary Triage)
   - Input: Retinal fundus scan (e.g. 512x512 or 224x224)
   - Classes:
       * 0: Non_Referable (Grade 0: No DR, Grade 1: Mild NPDR)
       * 1: Referable     (Grade 2: Moderate, Grade 3: Severe, Grade 4: Proliferative DR)
   - Decision Gate: Tunable referral threshold tau_referable (default 0.30)
     Calibrated to eliminate dangerous false negatives (sensitivity > 90-94%).

2. Stage 2: Severity Stager (Decoupled Specialist Classifiers)
   - Referable Stager (Stage 2B): Differentiates Grades 2, 3, 4
       * Moderate NPDR (Grade 2)
       * Severe NPDR (Grade 3)
       * Proliferative DR (Grade 4)
   - Non-Referable Stager (Stage 2A): Differentiates Grades 0, 1
       * No DR (Grade 0)
       * Mild NPDR (Grade 1)

Backbone Options (Edge-Optimized for Rural PHC mini-PCs / laptops):
  - ResNet-18 (default, 11.2M params, fast, clean Grad-CAM on layer4)
  - MobileNetV3-Small / Large (2.5M - 5.4M params, ultra-low power)
  - EfficientNet-B0 (5.3M params, high parameter efficiency)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger("visionx.two_stage")

CANONICAL_CLASSES: List[str] = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
STAGE1_CLASSES: List[str] = ["Non_Referable", "Referable"]
STAGE2_REF_CLASSES: List[str] = ["Moderate", "Severe", "Proliferate_DR"]
STAGE2_NONREF_CLASSES: List[str] = ["No_DR", "Mild"]

CLASS_TO_GRADE: Dict[str, int] = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}

GRADE_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_GRADE.items()}


def _build_backbone(
    backbone_name: str = "resnet18",
    pretrained: bool = False,
) -> Tuple[nn.Module, int, nn.Module]:
    """
    Builds a lightweight feature extractor backbone for PHC edge deployment.

    Returns:
        backbone_model: The underlying torchvision model
        in_features: Number of output feature channels before classification
        cam_target_layer: The last convolutional layer suitable for Grad-CAM
    """
    name = backbone_name.lower().replace("-", "_")

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base = models.resnet18(weights=weights)
        in_features = base.fc.in_features
        cam_layer = base.layer4[-1]
        base.fc = nn.Identity()
        return base, in_features, cam_layer

    elif name in ("mobilenet_v3_small", "mobilenetv3_small", "mobilenetv3"):
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        in_features = base.classifier[0].in_features
        cam_layer = base.features[-1]
        base.classifier = nn.Identity()
        return base, in_features, cam_layer

    elif name in ("mobilenet_v3_large", "mobilenetv3_large"):
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_large(weights=weights)
        in_features = base.classifier[0].in_features
        cam_layer = base.features[-1]
        base.classifier = nn.Identity()
        return base, in_features, cam_layer

    elif name in ("efficientnet_b0", "efficientnet"):
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base = models.efficientnet_b0(weights=weights)
        in_features = base.classifier[1].in_features
        cam_layer = base.features[-1]
        base.classifier = nn.Identity()
        return base, in_features, cam_layer

    else:
        raise ValueError(
            f"Unsupported backbone: '{backbone_name}'. "
            f"Choose from: 'resnet18', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_b0'."
        )


class Stage1ReferralGate(nn.Module):
    """
    Stage 1: Binary Referral Gatekeeper Network.

    Classifies fundus images into:
      - Class 0: Non_Referable (Grades 0 - 1) -> Routine annual review
      - Class 1: Referable     (Grades 2 - 4) -> Specialist referral required
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.30,
        tau_referable: float = 0.30,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.tau_referable = tau_referable
        self.base, in_features, self.cam_layer = _build_backbone(backbone, pretrained=pretrained)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2),  # [Non_Referable, Referable]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw binary logits [B, 2]."""
        feats = self.base(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        logits = self.classifier(feats)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities [B, 2]."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def predict_referable(
        self,
        x: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts referral decision using calibrated threshold tau.

        Args:
            x: Input tensor [B, 3, H, W]
            tau: Referral threshold. If None, uses self.tau_referable (default: 0.30)

        Returns:
            is_referable: Boolean tensor [B] (True if p_ref >= tau)
            p_referable: Float tensor [B] containing posterior probability of referable DR
        """
        threshold = self.tau_referable if tau is None else tau
        probs = self.predict_proba(x)
        p_ref = probs[:, 1]
        is_ref = p_ref >= threshold
        return is_ref, p_ref

    def get_cam_target_layer(self) -> nn.Module:
        """Returns target convolutional layer for Grad-CAM."""
        return self.cam_layer


class Stage2SeverityStager(nn.Module):
    """
    Stage 2: Specialist Disease Severity Stager Network.

    Contains two decoupled specialist heads:
      1. Referable Specialist (Grades 2, 3, 4):
         - Moderate NPDR (Grade 2)
         - Severe NPDR (Grade 3)
         - Proliferative DR (Grade 4)
      2. Non-Referable Specialist (Grades 0, 1):
         - No DR (Grade 0)
         - Mild NPDR (Grade 1)
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = False,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.base, in_features, self.cam_layer = _build_backbone(backbone, pretrained=pretrained)

        # Specialist Stager B: Referable (Grades 2, 3, 4)
        self.head_referable = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, 3),  # [Moderate, Severe, Proliferate_DR]
        )

        # Specialist Stager A: Non-Referable (Grades 0, 1)
        self.head_non_referable = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, 2),  # [No_DR, Mild]
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.base(x)
        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        return feats

    def forward_referable(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for [Moderate (2), Severe (3), Proliferate_DR (4)]."""
        feats = self.extract_features(x)
        return self.head_referable(feats)

    def forward_non_referable(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits for [No_DR (0), Mild (1)]."""
        feats = self.extract_features(x)
        return self.head_non_referable(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Default forward evaluates the primary Referable stager head (Grades 2, 3, 4).
        """
        return self.forward_referable(x)

    def get_cam_target_layer(self) -> nn.Module:
        return self.cam_layer


class VisionXTwoStagePipeline(nn.Module):
    """
    Complete Two-Stage Screening Pipeline coordinating Stage 1 and Stage 2.

    Mirrors matlab/inference/visionx_pipeline.m and matlab/evaluation/evaluate_two_stage.m:
      1. Stage 1 Referral Gate computes p(Referable).
      2. If p(Referable) >= tau_referable (calibrated clinical triage):
           Route to Stage 2B (Referable Stager) -> Grade 2, 3, or 4.
         Else:
           Route to Stage 2A (Non-Referable Stager) -> Grade 0 or 1.
    """

    def __init__(
        self,
        gate: Optional[Stage1ReferralGate] = None,
        stager: Optional[Stage2SeverityStager] = None,
        backbone: str = "resnet18",
        tau_referable: float = 0.30,
    ):
        super().__init__()
        self.gate = gate if gate is not None else Stage1ReferralGate(backbone=backbone, tau_referable=tau_referable)
        self.stager = stager if stager is not None else Stage2SeverityStager(backbone=backbone)
        self.tau_referable = tau_referable

    def forward(
        self,
        x: torch.Tensor,
        tau: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Runs full two-stage diagnostic inference for a batch or single image.

        Args:
            x: Input tensor [B, 3, H, W]
            tau: Optional triage threshold override (default: self.tau_referable)

        Returns:
            Dict containing:
              - 'grades': list of predicted ICDR integer grades [0..4]
              - 'labels': list of grade names (e.g. 'Moderate')
              - 'is_referable': list of bools
              - 'p_referable': list of float probabilities
              - 'confidences': list of float model confidences [0..1]
              - 'gate_probs': tensor of Stage 1 posterior probabilities
              - 'stager_probs': list of tensors containing stage 2 posteriors
        """
        threshold = self.tau_referable if tau is None else tau
        batch_size = x.size(0)

        # Stage 1: Referral Gatekeeper
        gate_logits = self.gate(x)
        gate_probs = F.softmax(gate_logits, dim=-1)
        p_ref = gate_probs[:, 1]
        is_ref_mask = p_ref >= threshold

        grades: List[int] = []
        labels: List[str] = []
        confidences: List[float] = []
        stager_probs_list: List[torch.Tensor] = []

        # Stage 2: Specialist Stagers
        for i in range(batch_size):
            img_tensor = x[i : i + 1]
            p_ref_val = float(p_ref[i].item())

            if is_ref_mask[i]:
                # Routed to Referable Stager (Grades 2, 3, 4)
                logits_ref = self.stager.forward_referable(img_tensor)
                probs_ref = F.softmax(logits_ref, dim=-1).squeeze(0)
                best_sub_idx = int(probs_ref.argmax().item())

                grade = 2 + best_sub_idx  # 0->2 (Moderate), 1->3 (Severe), 2->4 (Proliferative)
                label = STAGE2_REF_CLASSES[best_sub_idx]
                conf = float(probs_ref[best_sub_idx].item())
                stager_probs_list.append(probs_ref)

            else:
                # Routed to Non-Referable Stager (Grades 0, 1)
                logits_non = self.stager.forward_non_referable(img_tensor)
                probs_non = F.softmax(logits_non, dim=-1).squeeze(0)
                best_sub_idx = int(probs_non.argmax().item())

                grade = best_sub_idx  # 0->No_DR, 1->Mild
                label = STAGE2_NONREF_CLASSES[best_sub_idx]
                conf = float(probs_non[best_sub_idx].item())
                stager_probs_list.append(probs_non)

            grades.append(grade)
            labels.append(label)
            confidences.append(round(conf, 4))

        return {
            "grades": grades,
            "labels": labels,
            "is_referable": is_ref_mask.cpu().tolist(),
            "p_referable": [round(float(p.item()), 4) for p in p_ref],
            "confidences": confidences,
            "gate_probs": gate_probs,
            "stager_probs": stager_probs_list,
        }


# =========================================================================
# FACTORY BUILDER FUNCTIONS
# =========================================================================

def build_stage1_gate(
    backbone: str = "resnet18",
    pretrained: bool = False,
    tau_referable: float = 0.30,
) -> Stage1ReferralGate:
    """Builds the Stage 1 Binary Referral Gatekeeper."""
    return Stage1ReferralGate(backbone=backbone, pretrained=pretrained, tau_referable=tau_referable)


def build_stage2_stager(
    backbone: str = "resnet18",
    pretrained: bool = False,
) -> Stage2SeverityStager:
    """Builds the Stage 2 Specialist Severity Stager."""
    return Stage2SeverityStager(backbone=backbone, pretrained=pretrained)


def build_two_stage_pipeline(
    stage1_checkpoint: Union[str, Path, None] = None,
    stage2_checkpoint: Union[str, Path, None] = None,
    backbone: str = "resnet18",
    pretrained: bool = False,
    tau_referable: float = 0.30,
) -> VisionXTwoStagePipeline:
    """Builds the complete end-to-end Two-Stage Pipeline."""
    gate = build_stage1_gate(backbone=backbone, pretrained=pretrained, tau_referable=tau_referable)
    if stage1_checkpoint and Path(stage1_checkpoint).is_file():
        ckpt = torch.load(str(stage1_checkpoint), map_location="cpu")
        gate.load_state_dict(ckpt.get("state_dict", ckpt))
    stager = build_stage2_stager(backbone=backbone, pretrained=pretrained)
    if stage2_checkpoint and Path(stage2_checkpoint).is_file():
        ckpt = torch.load(str(stage2_checkpoint), map_location="cpu")
        stager.load_state_dict(ckpt.get("state_dict", ckpt))
    return VisionXTwoStagePipeline(gate=gate, stager=stager, backbone=backbone, tau_referable=tau_referable)

