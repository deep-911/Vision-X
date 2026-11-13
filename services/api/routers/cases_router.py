"""Sample Cases & Screening Router for VisionX."""

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np

from ..inference_service import (
    BENCHMARK_CASE_IDS,
    BENCHMARK_PROFILES,
    get_benchmark_image,
    run_full_pipeline,
    run_full_pipeline_async,
)
from ..schemas import AnalysisResponse

router = APIRouter(prefix="/cases", tags=["Cases & Screening"])


@router.get("", response_model=list[dict])
@router.get("/benchmark", response_model=list[dict])
async def get_cases():
    """Returns the list of sample retinal scans."""
    result = []
    for idx, case_id in enumerate(BENCHMARK_CASE_IDS):
        prof = BENCHMARK_PROFILES.get(case_id, {})
        pm = prof.get("patient_meta", {})
        result.append({
            "id": case_id,
            "display_name": f"Scan {idx + 1}",
            "grade": prof.get("grade", 0),
            "label": prof.get("label", "No_DR"),
            "patient_name": pm.get("patient_name", f"Patient {idx + 1}"),
            "laterality": pm.get("laterality", "OD"),
            "dataset": "APTOS 2019 (Aravind Eye Hospital Cohort)",
            "center": pm.get("phc_center", "Community Screening Clinic"),
            "is_referable": prof.get("is_referable", False),
            "csme_threat": prof.get("csme_threat", False),
            "min_fovea_dist_dd": prof.get("min_fovea_dist_dd"),
            "ex_clusters": prof.get("ex_clusters", 0),
            "he_foci": prof.get("he_foci", 0),
            "evidence": prof.get("evidence", ""),
            "confidence": prof.get("confidence", 90.0),
            "quality_status": "GRADEABLE_ACCEPT",
        })
    return result


@router.get("/analyze/{case_id}", response_model=AnalysisResponse)
async def analyze_benchmark_case(case_id: str):
    """Runs retinal image grading and lesion detection."""
    image_rgb = get_benchmark_image(case_id)
    if image_rgb is None:
        raise HTTPException(status_code=404, detail=f"Scan '{case_id}' not found.")

    meta = {
        "id": case_id,
        "dataset": "APTOS 2019 Indian Demographic Cohort",
        "benchmark": True,
    }
    result = await run_full_pipeline_async(image_rgb, meta)
    return result


@router.post("/upload", response_model=AnalysisResponse)
@router.post("/analyze", response_model=AnalysisResponse)
async def upload_and_analyze(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
):
    """Accepts a retinal photo, checks image quality, and runs screening analysis."""
    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="A retinal image file is required (field name 'image' or 'file').",
        )

    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    content = await upload.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds 25 MB limit.")

    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(status_code=400, detail="Could not read image file. Please use a PNG or JPEG photo.")

    image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    filename = upload.filename or "scan.png"
    meta = {
        "id": f"UPLOAD-{filename}",
        "filename": filename,
        "dataset": "Uploaded Retinal Scan",
        "benchmark": False,
    }
    result = await run_full_pipeline_async(image_rgb, meta)
    return result
