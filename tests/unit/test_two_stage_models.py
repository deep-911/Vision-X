"""
Unit tests for VisionX PyTorch Authentic Two-Stage Pipeline & Grad-CAM.
=======================================================================
Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
Team: Indian_Gryffindor
"""

import sys
from pathlib import Path
import numpy as np
import torch
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.two_stage import (
    CANONICAL_CLASSES,
    STAGE1_CLASSES,
    STAGE2_REF_CLASSES,
    STAGE2_NONREF_CLASSES,
    Stage1ReferralGate,
    Stage2SeverityStager,
    VisionXTwoStagePipeline,
    build_stage1_gate,
    build_stage2_stager,
    build_two_stage_pipeline,
)
from ml.models.gradcam import GradCAM
from ml.quality.image_quality import RetinalQualityAssessor


def test_stage1_referral_gate_forward_and_thresholding():
    """Verify Stage 1 Gate outputs binary logits and calibrated referral decision."""
    gate = build_stage1_gate(backbone="resnet18", pretrained=False, tau_referable=0.30)
    gate.eval()

    dummy_input = torch.randn(2, 3, 224, 224)
    logits = gate(dummy_input)

    assert logits.shape == (2, 2)
    probs = gate.predict_proba(dummy_input)
    assert probs.shape == (2, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2), atol=1e-5)

    is_ref, p_ref = gate.predict_referable(dummy_input, tau=0.30)
    assert is_ref.shape == (2,)
    assert p_ref.shape == (2,)
    assert isinstance(is_ref[0].item(), bool)


def test_stage2_severity_stager_heads():
    """Verify Stage 2 Stager decoupled specialist heads for Referable and Non-Referable."""
    stager = build_stage2_stager(backbone="resnet18", pretrained=False)
    stager.eval()

    dummy_input = torch.randn(2, 3, 224, 224)

    # Part A: Referable Stager (Moderate, Severe, PDR -> 3 classes)
    logits_ref = stager.forward_referable(dummy_input)
    assert logits_ref.shape == (2, 3)

    # Part B: Non-Referable Stager (No_DR, Mild -> 2 classes)
    logits_non = stager.forward_non_referable(dummy_input)
    assert logits_non.shape == (2, 2)


def test_two_stage_pipeline_end_to_end():
    """Verify full Two-Stage Pipeline coordinates Gatekeeper and Specialist Stagers."""
    pipeline = build_two_stage_pipeline(backbone="resnet18", pretrained=False, tau_referable=0.30)
    pipeline.eval()

    dummy_input = torch.randn(3, 3, 224, 224)
    result = pipeline(dummy_input)

    assert "grades" in result
    assert "labels" in result
    assert "is_referable" in result
    assert "p_referable" in result
    assert "confidences" in result

    assert len(result["grades"]) == 3
    for grade in result["grades"]:
        assert 0 <= grade <= 4

    for label in result["labels"]:
        assert label in CANONICAL_CLASSES


def test_gradcam_generation_and_overlay():
    """Verify Grad-CAM produces attention heatmaps and visual blending."""
    model = build_stage1_gate(backbone="resnet18", pretrained=False)
    cam = GradCAM(model, target_layer=model.get_cam_target_layer())

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = cam.generate_heatmap(dummy_input, target_class=1, target_size=(256, 256))

    assert isinstance(heatmap, np.ndarray)
    assert heatmap.shape == (256, 256)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

    dummy_fundus = np.zeros((256, 256, 3), dtype=np.uint8)
    dummy_fundus[:, :, 1] = 150  # green channel
    blended = GradCAM.overlay_heatmap(dummy_fundus, heatmap, alpha=0.40)

    assert blended.shape == (256, 256, 3)
    assert blended.dtype == np.uint8

    cam.remove_hooks()


def test_quality_assessor_usability_score_and_laplacian():
    """Verify Fundus Quality Assessor calculates 0-1 usability score and Laplacian variance."""
    assessor = RetinalQualityAssessor()
    sample_img = np.zeros((512, 512, 3), dtype=np.uint8)
    sample_img[50:450, 50:450] = [30, 100, 180]

    result = assessor.assess(sample_img)
    assert "usability_score" in result
    assert 0.0 <= result["usability_score"] <= 1.0
    assert "overall_usability" in result
    assert "laplacian_variance" in result["focus"]
    assert isinstance(result["focus"]["laplacian_variance"], (int, float))
