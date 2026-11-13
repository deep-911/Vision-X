"""Integration API Tests for VisionX using FastAPI TestClient."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from services.api.main import app


@pytest.fixture(scope="module")
def api_client():
    """Provides a module-scoped TestClient."""
    with TestClient(app) as c:
        yield c


def test_health_check(api_client):
    """Verify system health endpoint and model presence."""
    response = api_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models"]["two_stage_pipeline_present"] is True


def test_auth_login_and_me(api_client):
    """Verify doctor authentication and /auth/me profile."""
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@visionx.health", "password": "Password123!"}
    )
    assert login_resp.status_code == 200
    auth_data = login_resp.json()
    assert "access_token" in auth_data
    token = auth_data["access_token"]
    assert auth_data["role"] == "specialist"

    me_resp = api_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == "doctor@visionx.health"


def test_auth_quick_switch(api_client):
    """Verify quick persona switching for demonstration."""
    resp = api_client.post("/api/v1/auth/quick-switch/phc_worker")
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "phc_worker"


def test_benchmark_cases_retrieval(api_client):
    """Verify benchmark case list retrieval."""
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@visionx.health", "password": "Password123!"}
    )
    token = login_resp.json()["access_token"]

    cases_resp = api_client.get("/api/v1/cases", headers={"Authorization": f"Bearer {token}"})
    assert cases_resp.status_code == 200
    cases = cases_resp.json()
    assert len(cases) >= 5


def test_benchmark_case_analyze_grounding(api_client):
    """Verify clinical analysis on benchmark case."""
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@visionx.health", "password": "Password123!"}
    )
    token = login_resp.json()["access_token"]

    # Golden Benchmark Case 000c1434d8d7 (Moderate NPDR, Grade 2 with CSME threat)
    cid = "000c1434d8d7"
    resp = api_client.get(f"/api/v1/cases/analyze/{cid}", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    res = resp.json()
    assert res["grade"] == 2
    assert res["csme_threat"] is True
    assert res["is_referable"] is True


def test_review_submission_and_retrieval(api_client):
    """Verify clinical audit review trail submission and querying."""
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@visionx.health", "password": "Password123!"}
    )
    token = login_resp.json()["access_token"]

    review_payload = {
        "patient_id": "000c1434d8d7",
        "image_id": "000c1434d8d7",
        "ai_suggested_grade": 2,
        "clinician_confirmed_grade": 2,
        "agreed_with_ai": True,
        "clinical_notes": "Hard exudates circinate ring in foveal avascular margin confirmed.",
        "followup_plan": "Urgent Referral: Anti-VEGF / Laser Photocoagulation",
        "reviewer_name": "Dr. Ramesh Gupta, MS (Ophthalmology)",
        "reviewer_role": "specialist",
        "clinical_center": "District Hospital Eye Center, Nanded"
    }

    sub_resp = api_client.post("/api/v1/reviews", json=review_payload, headers={"Authorization": f"Bearer {token}"})
    assert sub_resp.status_code == 200
    assert sub_resp.json()["success"] is True

    get_resp = api_client.get("/api/v1/reviews?limit=5", headers={"Authorization": f"Bearer {token}"})
    assert get_resp.status_code == 200
    reviews = get_resp.json()
    assert isinstance(reviews, list)


def test_phc_intake_queue(api_client):
    """Verify PHC intake queue retrieval."""
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@visionx.health", "password": "Password123!"}
    )
    token = login_resp.json()["access_token"]

    resp = api_client.get("/api/v1/phc/intake?limit=5", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_bilateral_od_os_intake(api_client):
    """Verify bilateral (OD and OS) patient intake and max-grade aggregation."""
    login_resp = api_client.post(
        "/api/v1/auth/login",
        json={"email": "worker@visionx.health", "password": "Password123!"}
    )
    token = login_resp.json()["access_token"]

    from pathlib import Path
    sample_dir = Path(__file__).resolve().parents[2] / "data" / "demo_samples"
    od_path = sample_dir / "002c21358ce6.png"  # Normal / Grade 0
    os_path = sample_dir / "000c1434d8d7.png"  # Moderate / Grade 2 + CSME

    with open(od_path, "rb") as f_od, open(os_path, "rb") as f_os:
        files = {
            "file_od": ("od_scan.png", f_od.read(), "image/png"),
            "file_os": ("os_scan.png", f_os.read(), "image/png"),
        }
        data = {
            "patient_id": "TEST-BILATERAL-001",
            "full_name": "Ramesh Patil",
            "age": 58,
            "gender": "Male",
            "phc_center": "Baramati Community Clinic",
            "duration_diabetes_years": 8,
        }
        resp = api_client.post(
            "/api/v1/phc/intake",
            data=data,
            files=files,
            headers={"Authorization": f"Bearer {token}"},
        )

    assert resp.status_code == 200
    res = resp.json()
    assert res["patient_id"] == "TEST-BILATERAL-001"
    assert "OD" in res["eyes_screened"]
    assert "OS" in res["eyes_screened"]
    assert res["patient_level_grade"] >= 2  # Worse eye (OS has Moderate DR)
    assert res["csme_threat"] is True       # OS has CSME threat
    assert res["urgency"] == "HIGH"

