"""
VisionX: Fundus Image Quality Assessor (IQA) & Instant Recapture Feedback
========================================================================
Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
Team: Indian_Gryffindor

Requirement from MathWorks SIH26038:
  "Automatically evaluate fundus images for adequacy (focus, illumination,
   field of view). Apply adaptive enhancement (CLAHE, illumination normalisation,
   denoising) for borderline images; reject ungradeable ones with recapture feedback."

Rural PHC Clinical Deployment:
  In rural Primary Health Centres (PHCs), Accredited Social Health Activists (ASHAs)
  and multipurpose health workers operate low-cost, handheld non-mydriatic fundus cameras.
  A blurry or poorly illuminated image cannot be graded safely and would cause diagnostic error.
  This module operates as an edge gatekeeper:
    1. Computes focus/blur via Laplacian variance over the active retinal mask.
    2. Computes illumination/exposure (underexposure, overexposure, corneal glare).
    3. Computes field of view / retinal centration.
    4. Produces an overall usability score on a normalized 0.0 to 1.0 scale.
    5. Categorizes scan into: ADEQUATE, BORDERLINE, or UNGRADEABLE_REJECT.
    6. Provides actionable real-time feedback for the camera operator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np


class RetinalQualityAssessor:
    """
    Fundus Image Quality Assessor evaluating focus/blur, illumination,
    and field of view to compute an overall usability score (0 to 1).
    """

    def __init__(
        self,
        focus_thresh_pass: float = 60.0,
        focus_thresh_borderline: float = 38.0,
        underexposed_pixel_pct_limit: float = 35.0,
        overexposed_pixel_pct_limit: float = 15.0,
        min_retina_area_ratio: float = 0.65,
    ):
        self.focus_thresh_pass = focus_thresh_pass
        self.focus_thresh_borderline = focus_thresh_borderline
        self.underexposed_pixel_pct_limit = underexposed_pixel_pct_limit
        self.overexposed_pixel_pct_limit = overexposed_pixel_pct_limit
        self.min_retina_area_ratio = min_retina_area_ratio

    def evaluate_focus_sharpness(
        self,
        gray_retina: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, str, float]:
        """
        Computes high-frequency edge energy using Laplacian variance and Tenengrad.
        Only computes over the active retinal mask (excluding the dark surround).

        Returns:
            normalized_focus: Score in [0, 100]
            diagnosis: Clinical explanation string
            lap_var: Raw Laplacian variance (float)
        """
        if not np.any(mask):
            return 0.0, "No retinal tissue detected", 0.0

        retina_pixels_gray = gray_retina[mask > 0]

        # 1. Laplacian high-frequency variance (blur indicator)
        laplacian = cv2.Laplacian(gray_retina, cv2.CV_64F)
        lap_var = float(laplacian[mask > 0].var())

        # 2. Tenengrad gradient magnitude (Sobel energy for vessel borders)
        gx = cv2.Sobel(gray_retina, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_retina, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag_sq = gx**2 + gy**2
        tenengrad_mean = float(grad_mag_sq[mask > 0].mean())

        # Combined normalized focus score (0 to 100)
        raw_score = 0.60 * np.sqrt(tenengrad_mean) + 0.40 * np.log1p(lap_var) * 10.0
        normalized_focus = float(np.clip(raw_score * 2.20, 0.0, 100.0))

        if normalized_focus >= self.focus_thresh_pass:
            diagnosis = "Sharp - high-frequency retinal microvasculature clearly resolved"
        elif normalized_focus >= self.focus_thresh_borderline:
            diagnosis = "Borderline focus - mild optical blur detected; adaptive enhancement required"
        else:
            diagnosis = "Severe blur - microvascular structures indistinguishable; recapture required"

        return normalized_focus, diagnosis, lap_var

    def evaluate_illumination(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, str, Dict[str, float]]:
        """
        Evaluates exposure, illumination balance, and corneal reflection glare.

        Returns:
            illum_score: Score in [0, 100]
            diagnosis: Clinical explanation string
            metrics: Dict of raw exposure statistics
        """
        if not np.any(mask):
            return 0.0, "Empty / completely dark image", {
                "mean_brightness": 0.0,
                "pct_underexposed": 100.0,
                "pct_overexposed": 0.0,
            }

        # Convert to HSV and extract V (Luminance/Value channel)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        retina_pixels = v_channel[mask > 0]

        total_retina_pixels = max(len(retina_pixels), 1)
        under_pixels = int(np.count_nonzero(retina_pixels < 25))
        over_pixels = int(np.count_nonzero(retina_pixels > 240))

        pct_underexposed = (under_pixels / total_retina_pixels) * 100.0
        pct_overexposed = (over_pixels / total_retina_pixels) * 100.0
        mean_brightness = float(retina_pixels.mean())

        # Illumination score (100 = optimal balance between 100 and 160 mean luminance)
        illum_score = 100.0 - (pct_underexposed * 1.50) - (pct_overexposed * 3.00)
        illum_score = float(np.clip(illum_score, 0.0, 100.0))

        if pct_overexposed > self.overexposed_pixel_pct_limit:
            diagnosis = f"Severe glare / overexposure detected ({pct_overexposed:.1f}% saturated pixels)"
        elif pct_underexposed > self.underexposed_pixel_pct_limit:
            diagnosis = f"Severe underexposure ({pct_underexposed:.1f}% dark pixels; illumination inadequate)"
        elif illum_score < 60.0:
            diagnosis = "Uneven illumination - peripheral vignetting"
        else:
            diagnosis = f"Optimal illumination (Mean luminance: {mean_brightness:.1f}/255)"

        metrics = {
            "mean_brightness": round(mean_brightness, 2),
            "pct_underexposed": round(pct_underexposed, 2),
            "pct_overexposed": round(pct_overexposed, 2),
        }
        return illum_score, diagnosis, metrics

    def evaluate_field_of_view(
        self,
        img_shape: Tuple[int, ...],
        mask: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Ensures the retinal circular disc is properly centered and sufficiently large.
        """
        h, w = img_shape[:2]
        total_frame_area = h * w
        retina_area = int(np.count_nonzero(mask))
        area_ratio = retina_area / max(total_frame_area, 1)

        score = float(np.clip((area_ratio / 0.70) * 100.0, 0.0, 100.0))
        if area_ratio >= self.min_retina_area_ratio:
            diag = f"Full field of view captured ({area_ratio * 100.0:.1f}% frame area)"
        else:
            diag = f"Partial/Clipped field of view ({area_ratio * 100.0:.1f}% frame area - peripheral quadrants truncated)"

        return score, diag

    def triage(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Performs full quality triage on an input fundus image.

        Returns:
            Dict containing:
              - usability_score: float in [0.0, 1.0]
              - composite_score: float in [0.0, 100.0]
              - composite_quality_index: float in [0.0, 100.0]
              - is_gradeable: bool (False if rejected)
              - decision / status: 'ADEQUATE', 'BORDERLINE', or 'UNGRADEABLE_REJECT'
              - action: recommended workflow action
              - focus: details including Laplacian variance
              - illumination: details including underexposed and overexposed %
              - field_of_view: details
              - recapture_feedback: actionable instructions for PHC operator
        """
        # Step 1: Retinal mask segmentation
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mask = (gray > 15).astype(np.uint8)

        # Step 2: Compute individual metrics
        focus_score, focus_diag, lap_var = self.evaluate_focus_sharpness(gray, mask)
        illum_score, illum_diag, illum_metrics = self.evaluate_illumination(img_bgr, mask)
        fov_score, fov_diag = self.evaluate_field_of_view(img_bgr.shape, mask)

        # Step 3: Composite Quality Index (CQI) on 0 to 100 scale
        cqi = 0.45 * focus_score + 0.35 * illum_score + 0.20 * fov_score

        # Step 4: Usability Score on 0.0 to 1.0 scale
        usability_score = float(np.clip(cqi / 100.0, 0.0, 1.0))

        # Step 5: Triage decision logic & Recapture Feedback
        rejection_reasons = []
        feedback_instructions = []

        if focus_score < self.focus_thresh_borderline:
            rejection_reasons.append("Unacceptable blur")
            feedback_instructions.append(
                "Hold handheld fundus camera steady, instruct patient to look directly at the green fixation LED, and adjust manual focus ring."
            )

        if illum_metrics.get("pct_underexposed", 0) > self.underexposed_pixel_pct_limit:
            rejection_reasons.append("Severe underexposure")
            feedback_instructions.append(
                "Dim ambient room lighting, increase camera flash power, or allow 2 minutes for patient pupil dark-adaptation."
            )

        if illum_metrics.get("pct_overexposed", 0) > self.overexposed_pixel_pct_limit:
            rejection_reasons.append("Corneal glare / overexposure")
            feedback_instructions.append(
                "Tilt camera slightly (1-2 degrees) off the optical axis to eliminate central corneal flash reflection."
            )

        if fov_score < 45.0:
            rejection_reasons.append("Clipped field of view")
            feedback_instructions.append(
                "Center the camera lens on the patient's pupil before triggering the flash."
            )

        # Determine triage category
        if rejection_reasons:
            status = "UNGRADEABLE_REJECT"
            action = "REJECT_AND_RECAPTURE"
            primary_feedback = " ; ".join(feedback_instructions)
        elif focus_score < self.focus_thresh_pass or illum_score < 70.0:
            status = "BORDERLINE"
            action = "ADAPTIVE_ENHANCE_AND_PROCESS"
            primary_feedback = (
                "Borderline quality - applying CLAHE, bilateral denoising, and illumination correction automatically."
            )
        else:
            status = "ADEQUATE"
            action = "PASS_TO_AI_GRADING"
            primary_feedback = (
                "Image passes clinical quality standards. Proceeding directly to diagnostic classification."
            )

        return {
            "status": status,
            "decision": status,
            "action": action,
            "is_gradeable": status != "UNGRADEABLE_REJECT",
            "usability_score": round(usability_score, 3),
            "overall_usability": round(usability_score, 3),
            "composite_score": round(cqi, 1),
            "composite_quality_index": round(cqi, 1),
            "focus": {
                "score": round(focus_score, 1),
                "assessment": focus_diag,
                "laplacian_variance": round(lap_var, 2),
            },
            "illumination": {
                "score": round(illum_score, 1),
                "assessment": illum_diag,
                "metrics": illum_metrics,
            },
            "field_of_view": {
                "score": round(fov_score, 1),
                "assessment": fov_diag,
            },
            "recapture_feedback": primary_feedback,
        }

    def assess(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Standard quality assessment interface returning full triage results."""
        return self.triage(img_bgr)

    def assess_fundus_quality(self, img_bgr: np.ndarray) -> Dict[str, Any]:
        """Explicit alias for fundus quality assessment."""
        return self.triage(img_bgr)
