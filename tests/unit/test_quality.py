"""Unit tests for RetinalQualityAssessor in ml.common.quality.image_quality."""

import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.quality.image_quality import RetinalQualityAssessor


def test_sharp_image_higher_focus_than_blurred():
    """Verify that a high-contrast pattern gets a higher focus score than its blurred version."""
    assessor = RetinalQualityAssessor()
    
    # Create synthetic circular retina with high-frequency checkerboard pattern
    h, w = 512, 512
    sharp_img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 200, 255, -1)
    
    # Add checkerboard texture inside retina
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    pattern = ((xx // 8) % 2 == (yy // 8) % 2).astype(np.uint8) * 180 + 40
    sharp_img[:, :, 1] = np.where(mask > 0, pattern, 0)
    sharp_img[:, :, 0] = np.where(mask > 0, pattern // 2, 0)

    # Create blurred version
    blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), 10)

    res_sharp = assessor.assess(sharp_img)
    res_blurred = assessor.assess(blurred_img)

    assert res_sharp["focus"]["score"] > res_blurred["focus"]["score"]
    assert res_sharp["composite_score"] > res_blurred["composite_score"]
    assert res_sharp["composite_quality_index"] > res_blurred["composite_quality_index"]
    assert res_sharp["is_gradeable"] is True
    assert res_blurred["is_gradeable"] is False or res_blurred["decision"] != "ADEQUATE"


def test_dark_underexposed_image_fails_illumination():
    """Verify that a dark / underexposed image is flagged with low illumination score and underexposure."""
    assessor = RetinalQualityAssessor()
    
    dark_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(dark_img, (256, 256), 200, (20, 20, 20), -1)
    
    res = assessor.assess(dark_img)
    assert res["is_gradeable"] is False
    assert res["decision"] == "UNGRADEABLE_REJECT"
    assert res["status"] == "UNGRADEABLE_REJECT"
    assert res["action"] == "REJECT_AND_RECAPTURE"
    assert "underexposure" in res["recapture_feedback"].lower() or res["illumination"]["score"] < 40.0
    assert res["illumination"]["metrics"]["pct_underexposed"] > 35.0
    assert res["illumination"]["score"] < 40.0


def test_assess_returns_expected_schema():
    """Verify that assess() returns expected schema with composite_score, is_gradeable, and decision."""
    assessor = RetinalQualityAssessor()
    sample_img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(sample_img, (256, 256), 200, (180, 120, 40), -1)
    
    res = assessor.assess(sample_img)
    assert "composite_score" in res
    assert isinstance(res["composite_score"], (int, float))
    assert "is_gradeable" in res
    assert isinstance(res["is_gradeable"], bool)
    assert "decision" in res
    assert isinstance(res["decision"], str)
    assert res["decision"] in ("ADEQUATE", "BORDERLINE", "UNGRADEABLE_REJECT")
    
    # Also verify backward-compatible triage keys
    assert "status" in res
    assert "action" in res
    assert "composite_quality_index" in res
    assert "focus" in res
    assert "illumination" in res
    assert "field_of_view" in res
    assert "recapture_feedback" in res
