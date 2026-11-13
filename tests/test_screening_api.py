"""Tests for VisionX Screening API & Two-Stage Pipeline inference.

Verifies:
1. POST /api/v1/screening/analyze with field name 'image'
2. POST /api/v1/screening/analyze with field name 'file'
3. POST /api/v1/cases/upload with field name 'image' and 'file'
4. Comprehensive Two-Stage output payload conformance for Frontend:
   - quality: is_acceptable, passed, overall_score, blur_score, illumination_score, message
   - grade (0-4), label ('No_DR', 'Mild_NPDR', 'Moderate_NPDR', 'Severe_NPDR', 'Proliferative_DR')
   - stage1: is_referable, confidence, decision
   - stage2: grade, label, confidence (when referable)
   - is_referable, urgency, referral_recommendation
   - explainability: heatmap_base64, lesion_summary, evidence, lesions
   - heatmap_base64, structure (optic disc, fovea, blood vessels), telemedicine
"""

from pathlib import Path
import sys
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.api.main import app
from services.api.inference_service import get_benchmark_image


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _get_test_image_bytes(case_id: str = "000c1434d8d7") -> bytes:
    img = get_benchmark_image(case_id)
    assert img is not None, f"Benchmark scan {case_id} not found"
    success, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    assert success
    return buf.tobytes()


def test_screening_analyze_with_image_field(client: TestClient):
    """Verify POST /api/v1/screening/analyze accepts 'image' multipart field."""
    img_bytes = _get_test_image_bytes("000c1434d8d7")
    resp = client.post(
        "/api/v1/screening/analyze",
        files={"image": ("000c1434d8d7.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    # 1. Quality gate object
    assert "quality" in data
    q = data["quality"]
    assert "status" in q
    assert "is_acceptable" in q
    assert isinstance(q["is_acceptable"], bool)
    assert q["is_acceptable"] is True
    assert "passed" in q
    assert q["passed"] is True
    assert "overall_score" in q
    assert isinstance(q["overall_score"], (int, float))
    assert "blur_score" in q
    assert isinstance(q["blur_score"], (int, float))
    assert "illumination_score" in q
    assert isinstance(q["illumination_score"], (int, float))
    assert "message" in q
    assert isinstance(q["message"], str)

    # 2. Grade and Label
    assert "grade" in data
    assert data["grade"] == 2
    assert "label" in data
    assert data["label"] == "Moderate_NPDR"

    # 3. Stage 1 Gatekeeper
    assert "stage1" in data
    s1 = data["stage1"]
    assert isinstance(s1["is_referable"], bool)
    assert s1["is_referable"] is True
    assert isinstance(s1["confidence"], (int, float))
    assert s1["decision"] == "REFERABLE"

    # 4. Stage 2 Stager (present when referable)
    assert "stage2" in data
    s2 = data["stage2"]
    assert s2 is not None
    assert s2["grade"] == 2
    assert s2["label"] == "Moderate_NPDR"
    assert isinstance(s2["confidence"], (int, float))

    # 5. Referral & Urgency
    assert data["is_referable"] is True
    assert data["urgency"] in ("Routine", "Non-urgent", "Urgent", "Immediate")
    assert data["urgency"] == "Urgent"  # Grade 2 + CSME is Urgent
    assert "referral_recommendation" in data
    assert len(data["referral_recommendation"]) > 10

    # 6. Explainability
    assert "explainability" in data
    exp = data["explainability"]
    assert "heatmap_base64" in exp
    assert exp["heatmap_base64"].startswith("data:image/")
    assert "lesion_summary" in exp
    assert "evidence" in exp
    assert "lesions" in exp
    assert isinstance(exp["lesions"], list)

    # 7. Heatmap Base64 top-level
    assert "heatmap_base64" in data
    assert data["heatmap_base64"].startswith("data:image/")

    # 8. Anatomical Retinal Structure
    assert "structure" in data
    st = data["structure"]
    assert "optic_disc" in st
    assert "fovea" in st
    assert "blood_vessels" in st
    assert st["optic_disc"]["detected"] is True
    assert st["fovea"]["detected"] is True

    # 9. Telemedicine
    assert "telemedicine" in data
    tm = data["telemedicine"]
    assert "queue_priority" in tm
    assert isinstance(tm["queue_priority"], int)
    assert "recommended_turnaround_hours" in tm
    assert isinstance(tm["recommended_turnaround_hours"], int)

    # 10. Frontend identifiers
    assert "case_id" in data
    assert "original_image_url" in data
    assert data["original_image_url"].startswith("data:image/")


def test_screening_analyze_with_file_fallback(client: TestClient):
    """Verify POST /api/v1/screening/analyze also accepts fallback 'file' multipart field."""
    img_bytes = _get_test_image_bytes("002c21358ce6")
    resp = client.post(
        "/api/v1/screening/analyze",
        files={"file": ("002c21358ce6.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grade"] == 0
    assert data["label"] == "No_DR"
    assert data["is_referable"] is False
    assert data["stage1"]["is_referable"] is False
    assert data["stage1"]["decision"] == "NON_REFERABLE"
    assert data["stage2"] is None  # Non-referable -> stage2 is None
    assert data["urgency"] == "Routine"


def test_cases_upload_with_image_field(client: TestClient):
    """Verify POST /api/v1/cases/upload accepts 'image' field."""
    img_bytes = _get_test_image_bytes("0024cdab0c1e")
    resp = client.post(
        "/api/v1/cases/upload",
        files={"image": ("0024cdab0c1e.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grade"] == 1
    assert data["label"] == "Mild_NPDR"
    assert data["is_referable"] is False
    assert data["urgency"] == "Non-urgent"


def test_cases_upload_with_file_field(client: TestClient):
    """Verify POST /api/v1/cases/upload maintains backwards compatibility with 'file' field."""
    img_bytes = _get_test_image_bytes("0024cdab0c1e")
    resp = client.post(
        "/api/v1/cases/upload",
        files={"file": ("0024cdab0c1e.png", img_bytes, "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["grade"] == 1
    assert data["label"] == "Mild_NPDR"


def test_screening_analyze_missing_file_400(client: TestClient):
    """Verify 400 error when no image is provided."""
    resp = client.post("/api/v1/screening/analyze")
    assert resp.status_code == 400
