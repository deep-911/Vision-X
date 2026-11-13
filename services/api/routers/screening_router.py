"""Screening Router for VisionX (SIH26038 by Team Indian_Gryffindor).

Provides real-time inference endpoints for PHC screening workflows:
- POST /api/v1/screening/analyze
- Accepts multipart/form-data with field 'image' (or fallback 'file')
- Returns full clinical JSON including Quality Gate, Two-Stage Pipeline (Gatekeeper + Stager),
  Grad-CAM heatmap, IDRiD lesion segmentation, ETDRS CSME assessment, and Telemedicine triage.
"""

import time
import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
import numpy as np

from ..inference_service import run_full_pipeline_async
from ..schemas import AnalysisResponse

router = APIRouter(prefix="/screening", tags=["Screening"])


@router.post("/analyze", response_model=AnalysisResponse)
@router.post("", response_model=AnalysisResponse)
@router.post("/upload", response_model=AnalysisResponse)
async def analyze_screening_image(
    image: UploadFile | None = File(None),
    file: UploadFile | None = File(None),
):
    """Analyze a retinal fundus image for diabetic retinopathy and CSME.

    Accepts multipart/form-data with field name 'image' (or fallback 'file').
    Runs the full Two-Stage inference pipeline:
    - Stage 0: Quality Gate (blur, illumination, field of view)
    - Stage 1: Referable DR Gatekeeper (Binary referable triage)
    - Stage 2: DR Staging Classifier (ICDR Grade 0-4 severity)
    - Explainability: Grad-CAM attention heatmap & IDRiD lesion segmentation
    - CSME: ETDRS distance from hard exudates to fovea center
    - Telemedicine: rural PHC triage recommendation and referral guidance
    """
    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=400,
            detail="A retinal fundus image is required. Send multipart/form-data with field name 'image' or 'file'.",
        )

    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file must be a valid image (got '{upload.content_type}').",
        )

    content = await upload.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="The uploaded image file is empty.")

    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds maximum allowable size limit of 25 MB.",
        )

    decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise HTTPException(
            status_code=400,
            detail="Could not decode image file. Please upload a valid PNG or JPEG retinal fundus photograph.",
        )

    image_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    filename = upload.filename or "retinal_scan.png"
    clean_stem = filename.rsplit(".", 1)[0][:16] if filename else "scan"

    meta = {
        "id": f"SCREEN-{clean_stem}-{int(time.time()) % 10000:04d}",
        "filename": filename,
        "dataset": "PHC Retinal Screening",
        "benchmark": False,
    }

    result = await run_full_pipeline_async(image_rgb, meta)
    return result
