"""Pydantic V2 Models and Request/Response Schemas for VisionX."""

from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# --- Authentication Schemas ---
class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    name: str
    email: str


class UserOut(BaseModel):
    email: str
    name: str
    role: str
    center: str | None = None
    created_at: datetime | None = None


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str
    role: str = "phc_worker"  # 'specialist', 'phc_worker', 'patient'
    center: str | None = None


# --- Clinical Inference & Triage Schemas ---
class PatientMeta(BaseModel):
    id: str
    name: str | None = None
    dataset: str | None = "User-provided image"
    benchmark: bool = False
    phc: str | None = None


class AnalysisResponse(BaseModel):
    analysis_status: str
    grade: int | None = None
    grade_name: str
    confidence: float | None = None
    p_referable: float | None = None
    is_referable: bool | None = None
    is_ungradeable: bool = False
    evidence: str
    protocol: str
    cqi: float | None = None
    quality: dict[str, Any]
    # IDRiD Multi-Class Lesion Measurements
    total_lesions: int = 0
    ex_count: int = 0
    ex_clusters: int = 0
    he_count: int = 0
    he_foci: int = 0
    se_count: int = 0
    lesion_model_connected: bool = True
    lesion_model_name: str = "IDRiD Multi-Class Lesion U-Net"
    # CSME Foveal Distance Measurements
    od_coord: list[int] | None = None
    fovea_coord: list[int] | None = None
    laterality: str | None = "OD"
    min_fovea_dist_dd: float | None = None
    csme_threat: bool = False
    macular_threat: str
    macular_threat_hi: str
    # Base64 Overlays
    images: dict[str, str]
    patient_meta: dict[str, Any]
    # Two-Stage Pipeline & Screening Workspace Enhancements
    label: str | None = None
    stage1: dict[str, Any] | None = None
    stage2: dict[str, Any] | None = None
    urgency: str | None = None
    referral_recommendation: str | None = None
    explainability: dict[str, Any] | None = None
    heatmap_base64: str | None = None
    structure: dict[str, Any] | None = None
    telemedicine: dict[str, Any] | None = None
    case_id: str | None = None
    original_image_url: str | None = None
    gradcam: str | None = None
    analysis: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")


# --- Telemedicine Review & Sign-Off Schemas ---
class RecordReviewRequest(BaseModel):
    patient_id: str
    image_id: str | None = None
    ai_grade: int | None = None
    ai_suggested_grade: int | None = None
    ai_confidence: float | None = None
    ai_referable_risk: float | None = None
    csme_threat: bool = False
    min_fovea_dist_dd: float | None = None
    macular_threat: str | None = None
    triage_level: str | None = "REFERRAL"
    doctor_grade: int | None = None
    clinician_confirmed_grade: int | None = None
    is_overridden: bool = False
    agreed_with_ai: bool | None = None
    doctor_notes: str | None = None
    clinical_notes: str | None = None
    followup_plan: str | None = None
    reviewer_name: str | None = None
    reviewer_role: str | None = None
    clinical_center: str | None = None
    dispatch_status: str = "CONFIRMED"
    patient_meta: dict[str, Any] = Field(default_factory=dict)


class ReviewOut(BaseModel):
    id: str
    patient_id: str
    ai_grade: int | None
    doctor_grade: int | None
    is_overridden: bool
    triage_level: str
    dispatch_status: str
    doctor_notes: str | None
    created_at: datetime
    reviewed_by: str | None


# --- PHC Patient Intake Schemas ---
class PatientRegisterRequest(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    phc_center: str
    systolic_bp: int | None = None
    diastolic_bp: int | None = None
    hba1c: float | None = None
    duration_diabetes_years: int | None = None


class PatientOut(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    phc_center: str
    intake_date: datetime
    triage_status: str | None = "PENDING_SCAN"
    latest_grade: int | None = None
    latest_csme: bool | None = None
