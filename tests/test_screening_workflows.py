"""End-to-end clinical screening workflows test suite."""

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
from services.api.inference.model_inference import locate_anatomical_landmarks
from services.api.inference_service import get_benchmark_image, run_full_pipeline


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _get_auth_token(client: TestClient, email: str = "doctor@visionx.health") -> str:
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": "Password123!"})
    assert resp.status_code == 200, f"Login failed for {email}"
    return resp.json()["access_token"]


def test_normal_retina_screening(client: TestClient):
    resp = client.get("/api/v1/cases/analyze/002c21358ce6")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == 0
    assert data["csme_threat"] is False
    assert data["is_referable"] is False
    assert data["cqi"] >= 70.0


def test_mild_npdr_screening(client: TestClient):
    resp = client.get("/api/v1/cases/analyze/0024cdab0c1e")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == 1
    assert data["csme_threat"] is False
    assert data["is_referable"] is False


def test_moderate_npdr_with_csme(client: TestClient):
    resp = client.get("/api/v1/cases/analyze/000c1434d8d7")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == 2
    assert data["csme_threat"] is True
    assert data["is_referable"] is True
    assert data["min_fovea_dist_dd"] is not None and data["min_fovea_dist_dd"] <= 1.0


def test_severe_npdr_screening(client: TestClient):
    resp = client.get("/api/v1/cases/analyze/0104b032c141")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == 3
    assert data["is_referable"] is True


def test_proliferative_dr_screening(client: TestClient):
    resp = client.get("/api/v1/cases/analyze/001639a390f0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["grade"] == 4
    assert data["is_referable"] is True


def test_motion_blur_quality_rejection(client: TestClient):
    base_img = get_benchmark_image("002c21358ce6")
    assert base_img is not None
    blurred = cv2.GaussianBlur(base_img, (65, 65), 0)
    _, buf = cv2.imencode(".png", cv2.cvtColor(blurred, cv2.COLOR_RGB2BGR))

    resp = client.post(
        "/api/v1/cases/upload",
        files={"file": ("blurred_scan.png", buf.tobytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_status"] == "QUALITY_REJECTED"
    assert data["is_ungradeable"] is True
    assert data["grade"] is None


def test_underexposed_quality_rejection(client: TestClient):
    base_img = get_benchmark_image("002c21358ce6")
    assert base_img is not None
    dark = (base_img * 0.10).astype(np.uint8)
    _, buf = cv2.imencode(".png", cv2.cvtColor(dark, cv2.COLOR_RGB2BGR))

    resp = client.post(
        "/api/v1/cases/upload",
        files={"file": ("dark_scan.png", buf.tobytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_status"] == "QUALITY_REJECTED"
    assert data["is_ungradeable"] is True


def test_overexposed_quality_rejection(client: TestClient):
    bright = np.full((512, 512, 3), 245, dtype=np.uint8)
    _, buf = cv2.imencode(".png", cv2.cvtColor(bright, cv2.COLOR_RGB2BGR))

    resp = client.post(
        "/api/v1/cases/upload",
        files={"file": ("overexposed.png", buf.tobytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["analysis_status"] == "QUALITY_REJECTED"
    assert data["is_ungradeable"] is True


def test_glare_rejection_in_landmarking():
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(img, (256, 256), 200, (180, 100, 30), -1)
    # True Optic Disc at nasal left (x=130, y=256)
    cv2.circle(img, (130, 256), 35, (250, 220, 190), -1)
    # Intense corneal glare artifact at upper right
    cv2.circle(img, (380, 120), 18, (255, 255, 255), -1)

    lm = locate_anatomical_landmarks(img)
    od_x, od_y = lm["od_coord"]
    assert abs(od_x - 130) < 30, f"Optic disc misplaced at x={od_x}"
    assert lm["laterality"] == "OD"


def test_bilateral_asymmetric_screening(client: TestClient):
    token = _get_auth_token(client, "worker@visionx.health")
    sample_dir = PROJECT_ROOT / "data" / "demo_samples"
    od_bytes = (sample_dir / "002c21358ce6.png").read_bytes()
    os_bytes = (sample_dir / "000c1434d8d7.png").read_bytes()

    files = {
        "file_od": ("od_normal.png", od_bytes, "image/png"),
        "file_os": ("os_moderate.png", os_bytes, "image/png"),
    }
    data = {
        "patient_id": "PATIENT-ASYM-01",
        "full_name": "Suresh Gokhale",
        "age": 62,
        "gender": "Male",
        "phc_center": "Baramati Community Clinic",
        "duration_diabetes_years": 9,
    }
    resp = client.post("/api/v1/phc/intake", data=data, files=files, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    res = resp.json()
    assert res["patient_level_grade"] >= 2
    assert res["csme_threat"] is True
    assert res["urgency"] == "HIGH"
    assert "OD" in res["eyes_screened"] and "OS" in res["eyes_screened"]


def test_bilateral_symmetric_screening(client: TestClient):
    token = _get_auth_token(client, "worker@visionx.health")
    sample_dir = PROJECT_ROOT / "data" / "demo_samples"
    severe_bytes = (sample_dir / "000c1434d8d7.png").read_bytes()

    files = {
        "file_od": ("od_severe.png", severe_bytes, "image/png"),
        "file_os": ("os_severe.png", severe_bytes, "image/png"),
    }
    data = {
        "patient_id": "PATIENT-SYM-02",
        "full_name": "Meera Joshi",
        "age": 55,
        "gender": "Female",
        "phc_center": "Baramati Community Clinic",
        "duration_diabetes_years": 14,
    }
    resp = client.post("/api/v1/phc/intake", data=data, files=files, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    res = resp.json()
    assert res["patient_level_grade"] >= 2
    assert res["csme_threat"] is True
    assert res["urgency"] == "HIGH"
    assert res["triage_status"] == "URGENT_REFERRAL"


def test_low_confidence_safety_deferral():
    # Low-contrast noise image creates an uncertain / uncalibrated model state
    noisy = np.random.randint(60, 110, (512, 512, 3), dtype=np.uint8)
    res = run_full_pipeline(noisy, {"id": "NOISE-SAMPLE", "benchmark": False})
    assert res["analysis_status"] in ("QUALITY_REJECTED", "MODEL_REVIEW_REQUIRED_ESCALATED")


def test_clinical_evidence_disagreement_deferral():
    from services.api.inference_service import _execute_pipeline_core
    normal_img = get_benchmark_image("002c21358ce6")
    assert normal_img is not None
    res = _execute_pipeline_core(normal_img, {"id": "CUSTOM_SCAN", "benchmark": False})
    assert res["analysis_status"] in ("MODEL_REVIEW_REQUIRED", "MODEL_REVIEW_REQUIRED_ESCALATED")


def test_specialist_review_and_override(client: TestClient):
    token = _get_auth_token(client, "doctor@visionx.health")
    review_payload = {
        "patient_id": "PATIENT-ASYM-01",
        "image_id": "os_moderate.png",
        "ai_suggested_grade": 2,
        "clinician_confirmed_grade": 3,
        "agreed_with_ai": False,
        "clinical_notes": "Blot hemorrhages across all quadrants; upgrade to Grade 3.",
        "followup_plan": "Urgent specialist review.",
        "reviewer_name": "Dr. Ramesh Gupta, MS (Ophthalmology)",
        "reviewer_role": "specialist",
        "clinical_center": "District Hospital Eye Center, Nanded",
    }
    resp = client.post("/api/v1/reviews", json=review_payload, headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    res = resp.json()
    assert res["success"] is True

    reviews_resp = client.get("/api/v1/reviews?limit=5", headers={"Authorization": f"Bearer {token}"})
    assert reviews_resp.status_code == 200
    reviews = reviews_resp.json()
    matched = [r for r in reviews if r.get("patient_id") == "PATIENT-ASYM-01"]
    assert len(matched) > 0
    assert matched[0]["is_overridden"] is True
    assert matched[0]["doctor_grade"] == 3


def test_fhir_diagnostic_report_export(client: TestClient):
    token = _get_auth_token(client, "doctor@visionx.health")
    resp = client.get("/api/v1/fhir/diagnostic-report/PATIENT-ASYM-01", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    bundle = resp.json()
    assert bundle["resourceType"] == "DiagnosticReport"
    assert bundle["status"] == "final"
    assert "code" in bundle

