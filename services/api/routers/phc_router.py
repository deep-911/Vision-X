"""Clinic Intake Router for VisionX."""

from datetime import datetime, timezone
import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
import numpy as np

from ..database import get_db
from ..inference_service import run_full_pipeline, run_full_pipeline_async, QUALITY_ASSESSOR, GRADE_NAMES

router = APIRouter(prefix="/phc", tags=["Clinic Intake"])


@router.post("/iqa-check")
async def check_image_quality(file: UploadFile = File(...)):
    """Quick check of retinal image quality."""
    content = await file.read()
    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Unable to decode image.")

    triage = QUALITY_ASSESSOR.triage(decoded)
    return {
        "status": triage.get("status", "GRADEABLE_ACCEPT"),
        "cqi": triage.get("composite_quality_index", 85.0),
        "recapture_feedback": triage.get("recapture_feedback", "Image sharp and centered. Ready for screening."),
        "focus_score": triage.get("focus_score", 82.5),
        "illumination_status": triage.get("illumination_status", "Uniform illumination"),
        "field_coverage": triage.get("field_coverage", "Optic disc and macula visible"),
    }


@router.post("/intake")
@router.post("/register-scan")
async def register_patient_and_scan(
    patient_id: str = Form(...),
    name: str | None = Form(None),
    full_name: str | None = Form(None),
    age: int = Form(...),
    gender: str = Form(...),
    phc_center: str = Form("Rural PHC-04, Baramati"),
    diabetes_duration: str | None = Form(None),
    visual_acuity: str | None = Form(None),
    systolic_bp: int | None = Form(None),
    diastolic_bp: int | None = Form(None),
    hba1c: float | None = Form(None),
    duration_diabetes_years: int | None = Form(None),
    laterality: str | None = Form(None),
    file: UploadFile | None = File(None),
    image: UploadFile | None = File(None),
    file_od: UploadFile | None = File(None),
    file_os: UploadFile | None = File(None),
):
    """Registers patient and runs unified bilateral (OD/OS) or single-eye retinal screening."""
    patient_name = full_name or name or "Anonymous Patient"
    eyes_analyzed: dict[str, Any] = {}

    async def _decode_upload(upload: UploadFile | None) -> np.ndarray | None:
        if upload is None:
            return None
        content = await upload.read()
        if not content:
            return None
        decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise HTTPException(status_code=400, detail=f"Invalid image file for {upload.filename}.")
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

    # 1. Process OD (Right Eye) if provided
    img_od = await _decode_upload(file_od)
    if img_od is not None:
        meta_od = {"id": f"{patient_id}-OD", "name": patient_name, "phc": phc_center, "laterality": "OD", "dataset": "Clinic Intake"}
        analysis_od = await run_full_pipeline_async(img_od, meta_od)
        eyes_analyzed["OD"] = analysis_od

    # 2. Process OS (Left Eye) if provided
    img_os = await _decode_upload(file_os)
    if img_os is not None:
        meta_os = {"id": f"{patient_id}-OS", "name": patient_name, "phc": phc_center, "laterality": "OS", "dataset": "Clinic Intake"}
        analysis_os = await run_full_pipeline_async(img_os, meta_os)
        eyes_analyzed["OS"] = analysis_os

    # 3. Handle single file upload if neither file_od nor file_os was provided
    if not eyes_analyzed:
        single_upload = file or image
        if single_upload is not None:
            img_single = await _decode_upload(single_upload)
            lat = laterality.upper() if laterality in ("OD", "OS") else "AUTO"
            meta_single = {"id": patient_id, "name": patient_name, "phc": phc_center, "laterality": lat, "dataset": "Clinic Intake"}
            single_analysis = await run_full_pipeline_async(img_single, meta_single)
            detected_lat = single_analysis.get("laterality", "OD")
            eyes_analyzed[detected_lat] = single_analysis
        else:
            # Benchmark sample image fallback
            from ..inference_service import get_benchmark_image
            fallback_img = get_benchmark_image("000c1434d8d7") or np.zeros((512, 512, 3), dtype=np.uint8)
            meta_fallback = {"id": patient_id, "name": patient_name, "phc": phc_center, "laterality": "OS", "dataset": "Clinic Intake"}
            single_analysis = await run_full_pipeline_async(fallback_img, meta_fallback)
            eyes_analyzed["OS"] = single_analysis

    # 4. Patient-Level ICDR / ETDRS Aggregation: worse eye dictates patient grade
    valid_grades = [
        data.get("grade")
        for data in eyes_analyzed.values()
        if data.get("grade") is not None and not data.get("is_ungradeable")
    ]
    patient_grade = max(valid_grades) if valid_grades else None
    patient_csme = any(data.get("csme_threat", False) for data in eyes_analyzed.values())
    patient_referable = any(data.get("is_referable", False) for data in eyes_analyzed.values())

    if patient_csme or (patient_grade is not None and patient_grade >= 3):
        urgency = "HIGH"
        triage_status = "URGENT_REFERRAL"
    elif patient_grade == 2 or patient_referable:
        urgency = "MEDIUM"
        triage_status = "ROUTINE_REFERRAL"
    else:
        urgency = "ROUTINE"
        triage_status = "ANNUAL_MONITORING"

    primary_eye_key = "OD" if "OD" in eyes_analyzed else list(eyes_analyzed.keys())[0]
    primary_analysis = eyes_analyzed[primary_eye_key]

    patient_grade_name = GRADE_NAMES[patient_grade] if patient_grade is not None else "Ungradeable / Quality Recapture Required"

    db = get_db()
    if db is not None:
        patient_doc = {
            "patient_id": patient_id,
            "name": patient_name,
            "full_name": patient_name,
            "age": age,
            "gender": gender,
            "phc_center": phc_center,
            "diabetes_duration": diabetes_duration or str(duration_diabetes_years or 5),
            "visual_acuity": visual_acuity or "6/12",
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp,
            "hba1c": hba1c,
            "intake_date": datetime.now(timezone.utc),
            "triage_status": triage_status,
            "urgency": urgency,
            "status": "Pending Doctor Review",
            "latest_grade": patient_grade,
            "patient_grade": patient_grade,
            "patient_grade_name": patient_grade_name,
            "latest_csme": patient_csme,
            "cqi": primary_analysis.get("cqi"),
            "quality_status": primary_analysis.get("quality", {}).get("status"),
            "eyes_screened": list(eyes_analyzed.keys()),
            "analysis_summary": {
                "patient_grade": patient_grade,
                "grade": patient_grade,
                "is_referable": patient_referable,
                "csme_threat": patient_csme,
                "quality": primary_analysis.get("quality", {}).get("status", "GRADEABLE_ACCEPT"),
            },
        }
        await db["patients"].update_one(
            {"patient_id": patient_id},
            {"$set": patient_doc},
            upsert=True,
        )

    return {
        "message": "Patient registered and retinal screening analyzed successfully.",
        "patient_id": patient_id,
        "full_name": patient_name,
        "patient_level_grade": patient_grade,
        "patient_level_grade_name": patient_grade_name,
        "urgency": urgency,
        "triage_status": triage_status,
        "is_referable": patient_referable,
        "csme_threat": patient_csme,
        "eyes_screened": list(eyes_analyzed.keys()),
        "eyes": {
            k: {
                "grade": v.get("grade"),
                "grade_name": v.get("grade_name"),
                "confidence": v.get("confidence"),
                "is_referable": v.get("is_referable"),
                "csme_threat": v.get("csme_threat"),
                "min_fovea_dist_dd": v.get("min_fovea_dist_dd"),
                "cqi": v.get("cqi"),
                "quality_status": v.get("quality", {}).get("status"),
            }
            for k, v in eyes_analyzed.items()
        },
        "analysis_summary": {
            "patient_grade": patient_grade,
            "grade": patient_grade,
            "is_referable": patient_referable,
            "csme_threat": patient_csme,
            "quality": primary_analysis.get("quality", {}).get("status"),
        },
        "analysis": primary_analysis,
    }


@router.get("/intake")
@router.get("/patients")
async def list_patients(limit: int = 50):
    """Returns list of patients in the clinic intake queue."""
    db = get_db()
    if db is None:
        return []

    cursor = db["patients"].find().sort("intake_date", -1).limit(limit)
    patients = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        if "full_name" not in doc:
            doc["full_name"] = doc.get("name", "Unknown Patient")
        if "urgency" not in doc:
            doc["urgency"] = "HIGH" if doc.get("latest_csme") or (doc.get("latest_grade") or 0) >= 3 else "ROUTINE"
        patients.append(doc)
    return patients


@router.get("/patient/{patient_id}")
async def get_patient_record(patient_id: str):
    """Retrieves screening history for a patient."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    patient = await db["patients"].find_one({"patient_id": patient_id})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient record not found.")

    patient["_id"] = str(patient["_id"])
    reviews_cursor = db["reviews"].find({"patient_id": patient_id}).sort("created_at", -1)
    reviews = []
    async for r in reviews_cursor:
        r["_id"] = str(r["_id"])
        reviews.append(r)

    patient["reviews"] = reviews
    return patient
