"""Unit tests for ETDRS CSME Macular Threat detection in model_inference."""

import sys
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.api.inference.model_inference import detect_csme_threat


def test_no_exudates_returns_no_threat():
    """Verify that when no exudates are present, CSME threat is False and distance is None."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (180, 100, 30), -1)
    
    empty_mask = np.zeros((512, 512), dtype=bool)
    result = detect_csme_threat(img, ex_mask_512=empty_mask)
    
    assert result["csme_threat"] is False
    assert result["threat"] is False
    assert result["min_fovea_dist_dd"] is None
    assert result["min_distance_dd"] is None
    assert result["ex_count"] == 0


def test_exudate_inside_foveal_danger_zone():
    """Verify that exudates placed close to fovea center (within 1.0 DD) trigger CSME threat."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (180, 100, 30), -1)
    # Bright optic disc on the left (x=120, y=256)
    cv2.circle(img, (120, 256), 35, (255, 230, 200), -1)
    
    # Locate fovea center
    baseline = detect_csme_threat(img, ex_mask_512=np.zeros((512, 512), dtype=bool))
    fx, fy = baseline["fovea_coord"]
    dd = baseline["dd_pixels"]
    
    # Place a patch of exudates 25px away (< 1.0 DD, since DD=80px)
    ex_mask = np.zeros((512, 512), dtype=bool)
    cv2.circle(ex_mask.view(np.uint8), (fx + 25, fy), 8, 1, -1)
    
    result = detect_csme_threat(img, ex_mask_512=ex_mask)
    assert result["csme_threat"] is True
    assert result["threat"] is True
    assert result["min_fovea_dist_dd"] is not None
    assert result["min_fovea_dist_dd"] <= 1.0
    assert result["min_distance_dd"] <= 1.0


def test_exudate_outside_foveal_danger_zone():
    """Verify that exudates placed far (> 1.0 DD) from fovea do NOT trigger CSME threat."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (180, 100, 30), -1)
    # Bright optic disc on the left (x=120, y=256)
    cv2.circle(img, (120, 256), 35, (255, 230, 200), -1)
    
    baseline = detect_csme_threat(img, ex_mask_512=np.zeros((512, 512), dtype=bool))
    fx, fy = baseline["fovea_coord"]
    dd = baseline["dd_pixels"]
    
    # Place an exudate patch far away (140px away > 1.0 DD)
    ex_mask = np.zeros((512, 512), dtype=bool)
    cv2.circle(ex_mask.view(np.uint8), (fx, max(fy - 140, 60)), 8, 1, -1)
    
    result = detect_csme_threat(img, ex_mask_512=ex_mask)
    assert result["csme_threat"] is False
    assert result["threat"] is False
    assert result["min_fovea_dist_dd"] is not None
    assert result["min_fovea_dist_dd"] > 1.0
    assert result["min_distance_dd"] > 1.0


def test_glare_artifact_rejected_for_optic_disc():
    """Verify that a saturated glare spot at the border does not hijack optic disc detection."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Retinal field
    cv2.circle(img, (256, 256), 200, (180, 100, 30), -1)
    # True Optic Disc on nasal side
    cv2.circle(img, (130, 256), 35, (250, 220, 190), -1)
    # Fake intense corneal glare artifact near the edge
    cv2.circle(img, (380, 120), 15, (255, 255, 255), -1)

    result = detect_csme_threat(img, ex_mask_512=np.zeros((512, 512), dtype=bool))
    od_x, od_y = result["od_coord"]
    # Should locate the true optic disc near x=130, not the glare at x=380
    assert abs(od_x - 130) < 30
    assert result["laterality"] == "OD"

