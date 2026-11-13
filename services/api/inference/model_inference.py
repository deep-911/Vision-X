"""Canonical server-side inference adapter for VisionX Two-Stage Retinal Screening Pipeline.

Supports:
1. Fast in-memory inference with Two-Stage Pipeline (Stage 1 Referral Gate + Stage 2 Severity Stager) & Grad-CAM (<300ms).
2. Edge-friendly backbones (ResNet-18 / MobileNetV3) suitable for rural PHC deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelUnavailable(RuntimeError):
    """Raised when a real model result cannot be produced."""


@dataclass(frozen=True)
class ModelResult:
    label: str
    confidence: float
    referable_risk: float
    attention_map: np.ndarray | None
    engine: str = "two_stage_pipeline"


CANONICAL_CLASSES = ["No_DR", "Mild", "Moderate", "Severe", "Proliferate_DR"]
CLASS_TO_GRADE = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe": 3,
    "Proliferate_DR": 4,
}

_TWO_STAGE_PIPELINE_CACHE = None
_TWO_STAGE_CAM_CACHE = None
_LESION_UNET_CACHE = None


def get_lesion_unet(model_path: Path | None = None) -> nn.Module | None:
    """Loads and caches the PyTorch IDRiD Multi-Class Lesion U-Net."""
    global _LESION_UNET_CACHE
    if not TORCH_AVAILABLE:
        return None
    if _LESION_UNET_CACHE is not None:
        return _LESION_UNET_CACHE

    if model_path is None:
        candidate = Path(__file__).resolve().parent.parent.parent.parent / "models" / "production" / "lesion_unet.pt"
        if candidate.is_file():
            model_path = candidate

    if model_path is None or not Path(model_path).is_file():
        return None

    try:
        project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from ml.models.lesion_unet import LesionUNet
        net = LesionUNet(in_channels=3, num_classes=4, base_filters=32)
        ckpt = torch.load(str(model_path), map_location="cpu")
        if "model_state_dict" in ckpt:
            net.load_state_dict(ckpt["model_state_dict"])
        else:
            net.load_state_dict(ckpt)
        net.eval()
        _LESION_UNET_CACHE = net
        return _LESION_UNET_CACHE
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception as e:
        return None


def locate_anatomical_landmarks(img_rgb: np.ndarray) -> dict:
    """Optic Disc & Fovea localization using vessel convergence and geometry.

    Rejects corneal glare, edge artifacts, and bright exudates by combining:
      1. Green-channel vessel convergence via morphological black top-hat filtering.
      2. Multi-spectral luminance (Red + Green channel agreement).
      3. Glare/reflection penalty (suppresses high-brightness zero-vesselness blobs).
      4. Temporal fovea dark depression search constrained by anatomical priors.
      5. Automatic eye laterality deduction (OD = Right Eye, OS = Left Eye).
    """
    h, w = 512, 512
    img = cv2.resize(img_rgb, (w, h)) if (img_rgb.shape[0] != h or img_rgb.shape[1] != w) else img_rgb
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Retinal field mask (with safe erosion to avoid perimeter glare)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)))

    # Green channel vessel convergence
    green = img[:, :, 1]
    vessels = cv2.morphologyEx(
        cv2.bitwise_not(green),
        cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
    )
    vessel_density = cv2.GaussianBlur(vessels.astype(np.float32), (41, 41), 0)
    v_diff = vessel_density.max() - vessel_density.min()
    v_norm = (
        (vessel_density - vessel_density.min()) / (v_diff + 1e-6)
        if v_diff > 1e-3
        else np.zeros((h, w), dtype=np.float32)
    )

    # Red & Green luminance
    r_blur = cv2.GaussianBlur(img[:, :, 0].astype(np.float32), (35, 35), 0)
    r_diff = r_blur.max() - r_blur.min()
    r_norm = (
        (r_blur - r_blur.min()) / (r_diff + 1e-6)
        if r_diff > 1e-3
        else np.zeros((h, w), dtype=np.float32)
    )

    g_blur = cv2.GaussianBlur(green.astype(np.float32), (35, 35), 0)
    g_diff = g_blur.max() - g_blur.min()
    g_norm = (
        (g_blur - g_blur.min()) / (g_diff + 1e-6)
        if g_diff > 1e-3
        else np.zeros((h, w), dtype=np.float32)
    )

    # Glare penalty: extreme brightness (> 240 in all channels) with near-zero vesselness
    glare_penalty = np.zeros((h, w), dtype=np.float32)
    bright_glare = (img[:, :, 0] > 240) & (img[:, :, 1] > 235) & (img[:, :, 2] > 220)
    if np.any(bright_glare):
        glare_blur = cv2.GaussianBlur(bright_glare.astype(np.float32), (31, 31), 0)
        glare_penalty = np.where(v_norm < 0.25, glare_blur * 0.5, 0.0)

    # Dynamic weighting: If real vessel patterns exist, weight vessel convergence heavily (40%)
    weight_v = 0.40 if v_diff > 5.0 else 0.0
    weight_r = 0.55 if weight_v == 0.0 else 0.35
    weight_g = 0.45 if weight_v == 0.0 else 0.25

    od_score = (weight_r * r_norm + weight_g * g_norm + weight_v * v_norm - glare_penalty) * (mask > 0)
    _, _, _, max_loc = cv2.minMaxLoc(od_score)
    od_x, od_y = max_loc
    dd = 80  # Canonical Disc Diameter in 512x512 pixels

    # Mask out Optic Disc (radius ~68 px)
    od_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(od_mask, (od_x, od_y), 68, 255, -1)

    # Laterality from disc position:
    # Right Eye (OD): Optic disc is nasal (left side of image, x < 256) -> Fovea is temporal (to the right)
    # Left Eye (OS): Optic disc is nasal (right side of image, x >= 256) -> Fovea is temporal (to the left)
    laterality = "OD" if od_x < 256 else "OS"

    if od_x < 256:
        cx_min = int(od_x + 1.6 * dd)
        cx_max = min(int(od_x + 2.8 * dd), 460)
    else:
        cx_min = max(int(od_x - 2.8 * dd), 50)
        cx_max = int(od_x - 1.6 * dd)

    cy_min = max(int(od_y - 0.7 * dd), 70)
    cy_max = min(int(od_y + 0.8 * dd), 440)

    search_mask = np.zeros((h, w), dtype=np.uint8)
    if cx_max > cx_min and cy_max > cy_min:
        search_mask[cy_min:cy_max, cx_min:cx_max] = 255
        search_mask = cv2.bitwise_and(search_mask, mask)
        masked_g = np.where(search_mask > 0, g_blur, 255.0)
        _, _, min_loc, _ = cv2.minMaxLoc(masked_g)
        fovea_x, fovea_y = min_loc
    else:
        fovea_x = min(int(od_x + 2.2 * dd), 420) if od_x < 256 else max(int(od_x - 2.2 * dd), 90)
        fovea_y = int(np.clip(od_y + 10, 200, 310))

    return {
        "od_coord": (int(od_x), int(od_y)),
        "fovea_coord": (int(fovea_x), int(fovea_y)),
        "laterality": laterality,
        "dd_pixels": dd,
        "od_mask": od_mask,
        "retinal_mask": mask,
    }


def segment_retinal_lesions(
    image_rgb: np.ndarray,
    model_path: Path | None = None,
    threshold_ex: float = 0.38,
    threshold_he: float = 0.25,
    threshold_se: float = 0.32,
) -> dict:
    """Runs Multi-Class Lesion U-Net on a fundus photograph.

    Extracts pixel masks and returns a color-coded transparent lesion overlay:
      - Hard Exudates (EX): Warm Amber/Yellow [245, 158, 11]
      - Hemorrhages & MAs (HE): Crimson Red [239, 68, 68]
      - Soft Exudates (SE): Cyan [6, 182, 212]
    """
    h_orig, w_orig = image_rgb.shape[:2]
    h_proc, w_proc = 512, 512
    img_proc = cv2.resize(image_rgb, (w_proc, h_proc))

    net = get_lesion_unet(model_path)

    # Retinal field mask and optic disc localization
    landmarks = locate_anatomical_landmarks(img_proc)
    mask = landmarks["retinal_mask"]
    od_mask = landmarks["od_mask"]

    if net is not None and TORCH_AVAILABLE:
        tensor = torch.from_numpy(img_proc).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0)

        with torch.no_grad():
            logits = net(tensor)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        ma_p = probs[0]
        he_p = probs[1]
        ex_p = probs[2]
        se_p = probs[3]

        ex_mask = (ex_p >= threshold_ex) & (mask > 0) & (od_mask == 0)
        he_mask = ((he_p >= threshold_he) | (ma_p >= 0.30)) & (mask > 0) & (od_mask == 0)
        se_mask = (se_p >= threshold_se) & (mask > 0) & (od_mask == 0)
        model_name = "VisionX IDRiD U-Net (PyTorch 512x512)"
        model_active = True
    else:
        # Fallback to morphological green-channel thresholding
        ex_mask = (img_proc[:, :, 1] > 200) & (img_proc[:, :, 0] > 200) & (mask > 0) & (od_mask == 0)
        he_mask = np.zeros((h_proc, w_proc), dtype=bool)
        se_mask = np.zeros((h_proc, w_proc), dtype=bool)
        model_name = "Morphological Color Filter (Heuristic Fallback)"
        model_active = False

    def _filter_min_area(mask_bool: np.ndarray, min_area: int = 20) -> np.ndarray:
        if not np.any(mask_bool):
            return mask_bool
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bool.astype(np.uint8))
        clean_mask = np.zeros_like(mask_bool)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_mask[labels == i] = True
        return clean_mask

    ex_mask = _filter_min_area(ex_mask, min_area=15)
    he_mask = _filter_min_area(he_mask, min_area=25)
    se_mask = _filter_min_area(se_mask, min_area=25)

    ex_count = int(np.sum(ex_mask))
    he_count = int(np.sum(he_mask))
    se_count = int(np.sum(se_mask))

    # Connected component analysis for lesion clustering
    num_ex_labels, _, _, _ = cv2.connectedComponentsWithStats(ex_mask.astype(np.uint8))
    ex_clusters = max(0, int(num_ex_labels) - 1)
    num_he_labels, _, _, _ = cv2.connectedComponentsWithStats(he_mask.astype(np.uint8))
    he_foci = max(0, int(num_he_labels) - 1)

    # Construct color-coded visual overlay
    overlay = img_proc.copy()
    overlay[he_mask] = [239, 68, 68]    # Hemorrhages: Crimson Red
    overlay[ex_mask] = [245, 158, 11]   # Hard Exudates: Warm Amber/Yellow
    overlay[se_mask] = [6, 182, 212]    # Soft Exudates: Cyan

    blended = cv2.addWeighted(img_proc, 0.65, overlay, 0.35, 0)

    # Draw crisp 1px boundary contours around lesion clusters
    contours_ex, _ = cv2.findContours(ex_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours_ex, -1, (245, 158, 11), 1)

    contours_he, _ = cv2.findContours(he_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours_he, -1, (239, 68, 68), 1)

    blended_orig = cv2.resize(blended, (w_orig, h_orig))

    return {
        "overlay_rgb": blended_orig,
        "ex_count": ex_count,
        "ex_clusters": ex_clusters,
        "he_count": he_count,
        "he_foci": he_foci,
        "se_count": se_count,
        "total_lesions": ex_count + he_count + se_count,
        "ex_mask_512": ex_mask,
        "he_mask_512": he_mask,
        "model_connected": model_active,
        "model_name": model_name,
    }


def detect_csme_threat(image_rgb: np.ndarray, ex_mask_512: np.ndarray | None = None) -> dict:
    """Evaluate Clinically Significant Macular Edema (CSME) via the ETDRS 1 DD rule.

    Calculates the Euclidean distance from hard exudate clusters to the fovea center.
    If min distance <= 1.0 Disc Diameter (DD), CSME threat is detected.
    """
    h, w = 512, 512
    img = cv2.resize(image_rgb, (w, h))

    # 1. Locate anatomical landmarks via hybrid vessel convergence & geometry
    landmarks = locate_anatomical_landmarks(img)
    od_x, od_y = landmarks["od_coord"]
    fovea_x, fovea_y = landmarks["fovea_coord"]
    dd = landmarks["dd_pixels"]
    mask = landmarks["retinal_mask"]
    od_mask = landmarks["od_mask"]
    laterality = landmarks["laterality"]

    # 2. Detect Hard Exudates (Use neural network U-Net mask if provided; fallback to color)
    if ex_mask_512 is not None:
        ex_y, ex_x = np.where(ex_mask_512)
        ex_count = int(len(ex_x))
    else:
        ex = (img[:, :, 1] > 200) & (img[:, :, 0] > 200) & (mask > 0) & (od_mask == 0)
        ex_y, ex_x = np.where(ex)
        ex_count = int(len(ex_x))

    if ex_count > 25:
        dists = np.sqrt((ex_x - fovea_x) ** 2 + (ex_y - fovea_y) ** 2)
        min_dd = float(np.min(dists)) / dd
        threat = bool(min_dd <= 1.0)
    else:
        min_dd = None
        threat = False

    status_label = "CSME Threat Present (High Risk)" if threat else "Macula Preserved (Low Risk)"
    status_label_hi = "सीएसएमई खतरा मौजूद (उच्च जोखिम)" if threat else "मैकुला सुरक्षित (कम जोखिम)"

    dist_val = round(min_dd, 2) if min_dd is not None else None

    return {
        "od_coord": [int(od_x), int(od_y)],
        "fovea_coord": [int(fovea_x), int(fovea_y)],
        "laterality": laterality,
        "dd_pixels": dd,
        "ex_count": ex_count,
        "min_fovea_dist_dd": dist_val,
        "min_distance_dd": dist_val,
        "csme_threat": threat,
        "threat": threat,
        "macular_threat": status_label,
        "macular_threat_hi": status_label_hi,
    }


def get_two_stage_pipeline(
    stage1_path: Path | None = None,
    stage2_path: Path | None = None,
    backbone: str = "resnet18",
    tau_referable: float = 0.30,
):
    """Loads and caches the authentic VisionX Two-Stage Pipeline."""
    global _TWO_STAGE_PIPELINE_CACHE, _TWO_STAGE_CAM_CACHE
    if _TWO_STAGE_PIPELINE_CACHE is not None:
        return _TWO_STAGE_PIPELINE_CACHE, _TWO_STAGE_CAM_CACHE

    if not TORCH_AVAILABLE:
        raise ModelUnavailable("PyTorch is not available for neural inference.")

    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from ml.models.two_stage import build_two_stage_pipeline
        from ml.models.gradcam import GradCAM

        pipeline = build_two_stage_pipeline(
            stage1_checkpoint=stage1_path if (stage1_path and Path(stage1_path).is_file()) else None,
            stage2_checkpoint=stage2_path if (stage2_path and Path(stage2_path).is_file()) else None,
            backbone=backbone,
            pretrained=False,
            tau_referable=tau_referable,
        )
        pipeline.eval()

        cam_target = pipeline.stager.get_cam_target_layer()
        cam = GradCAM(pipeline.stager, cam_target)

        _TWO_STAGE_PIPELINE_CACHE = pipeline
        _TWO_STAGE_CAM_CACHE = cam
        return _TWO_STAGE_PIPELINE_CACHE, _TWO_STAGE_CAM_CACHE
    except Exception as exc:
        raise ModelUnavailable(f"Failed to initialize Two-Stage Pipeline: {exc}") from exc


def infer_with_two_stage(
    image_rgb: np.ndarray,
    *,
    stage1_path: Path | None = None,
    stage2_path: Path | None = None,
    backbone: str = "resnet18",
    tau_referable: float = 0.30,
) -> ModelResult:
    """Run two-stage screening inference and Grad-CAM in memory."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ModelUnavailable("Only decoded RGB images can be classified.")

    pipeline, cam = get_two_stage_pipeline(
        stage1_path=stage1_path,
        stage2_path=stage2_path,
        backbone=backbone,
        tau_referable=tau_referable,
    )

    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm,
    ])

    input_tensor = eval_transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        out = pipeline(input_tensor, tau=tau_referable)

    label = out["labels"][0]
    confidence = float(out["confidences"][0])
    p_ref = float(out["p_referable"][0])

    try:
        input_cam = eval_transform(image_rgb).unsqueeze(0)
        grade_val = CLASS_TO_GRADE.get(label, 0)
        stager_idx = max(0, min(2, grade_val - 2)) if grade_val >= 2 else max(0, min(1, grade_val))
        cam_map = cam.generate(input_cam, target_class=stager_idx)
        if cam_map is not None:
            cam_map_resized = cv2.resize(cam_map, (512, 512))
            attention_map = (cam_map_resized * 255).astype(np.uint8)
        else:
            attention_map = None
    except Exception:
        attention_map = None

    return ModelResult(
        label=label,
        confidence=confidence,
        referable_risk=p_ref,
        attention_map=attention_map,
        engine="two_stage_pipeline",
    )


def infer_scan(
    image_rgb: np.ndarray,
    *,
    stage1_path: Path | None = None,
    stage2_path: Path | None = None,
    backbone: str = "resnet18",
    prefer_fast_engine: bool = True,
    timeout_seconds: int = 90,
    **kwargs,
) -> ModelResult:
    """Unified entrypoint: executes in-memory Two-Stage Screening Pipeline."""
    return infer_with_two_stage(
        image_rgb,
        stage1_path=stage1_path,
        stage2_path=stage2_path,
        backbone=backbone,
    )
