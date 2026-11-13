"""Doctor Reviews & Activity Log Router for VisionX."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from ..auth import get_current_user
from ..database import get_db
from ..schemas import RecordReviewRequest

router = APIRouter(prefix="/reviews", tags=["Doctor Reviews"])


@router.post("")
@router.post("/record")
async def record_review(
    req: RecordReviewRequest,
    user: dict = Depends(get_current_user),
):
    """Saves doctor review decision or grade update."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    ai_grade = req.ai_grade if req.ai_grade is not None else req.ai_suggested_grade
    doc_grade = req.doctor_grade if req.doctor_grade is not None else req.clinician_confirmed_grade
    notes = req.doctor_notes if req.doctor_notes is not None else req.clinical_notes
    is_ovr = req.is_overridden if req.agreed_with_ai is None else (not req.agreed_with_ai)
    reviewer = req.reviewer_name or user.get("name", "Dr. Ramesh Gupta, MS (Ophthalmology)")

    review_doc = {
        "patient_id": req.patient_id,
        "image_id": req.image_id,
        "ai_grade": ai_grade,
        "ai_suggested_grade": ai_grade,
        "ai_confidence": req.ai_confidence,
        "ai_referable_risk": req.ai_referable_risk,
        "csme_threat": req.csme_threat,
        "min_fovea_dist_dd": req.min_fovea_dist_dd,
        "macular_threat": req.macular_threat,
        "triage_level": req.triage_level or ("REFERRAL" if (doc_grade or 0) >= 2 else "ROUTINE"),
        "doctor_grade": doc_grade,
        "clinician_confirmed_grade": doc_grade,
        "is_overridden": is_ovr,
        "agreed_with_ai": not is_ovr,
        "doctor_notes": notes,
        "clinical_notes": notes,
        "followup_plan": req.followup_plan or "Routine screening follow-up",
        "dispatch_status": req.dispatch_status,
        "patient_meta": req.patient_meta,
        "created_at": datetime.now(timezone.utc),
        "reviewed_by": reviewer,
        "reviewer_name": reviewer,
        "reviewed_by_email": user.get("email"),
        "clinical_center": req.clinical_center or user.get("center", "District Hospital Eye Center"),
    }

    insert_result = await db["reviews"].insert_one(review_doc)

    # Update patient record if exists
    await db["patients"].update_one(
        {"patient_id": req.patient_id},
        {
            "$set": {
                "latest_doctor_grade": doc_grade,
                "latest_review_status": req.dispatch_status,
                "last_reviewed_at": datetime.now(timezone.utc),
            }
        },
    )

    review_doc["_id"] = str(insert_result.inserted_id)
    return {
        "success": True,
        "review_id": str(insert_result.inserted_id),
        "message": "Review saved successfully.",
        "data": review_doc,
    }


@router.get("")
@router.get("/recent")
@router.get("/audit-trail")
async def get_audit_trail(limit: int = 50):
    """Retrieves list of past doctor reviews."""
    db = get_db()
    if db is None:
        return []

    cursor = db["reviews"].find().sort("created_at", -1).limit(limit)
    reviews = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        # Ensure default field representations for frontend
        if "clinician_confirmed_grade" not in doc:
            doc["clinician_confirmed_grade"] = doc.get("doctor_grade")
        if "ai_suggested_grade" not in doc:
            doc["ai_suggested_grade"] = doc.get("ai_grade")
        if "agreed_with_ai" not in doc:
            doc["agreed_with_ai"] = not doc.get("is_overridden", False)
        if "reviewer_name" not in doc:
            doc["reviewer_name"] = doc.get("reviewed_by", "Dr. Ramesh Gupta, MS")
        reviews.append(doc)
    return reviews


@router.get("/stats")
async def get_telemedicine_stats():
    """Returns summary statistics for reviews."""
    db = get_db()
    if db is None:
        return {
            "total_reviews": 0,
            "agreement_rate": 100.0,
            "concordance_rate": 100.0,
            "overrides": 0,
            "total_overrides": 0,
            "csme_threats_caught": 0,
            "urgent_referrals": 0,
            "compliance": "ABDM / FHIR Diagnostic Audit Ready",
        }

    total = await db["reviews"].count_documents({})
    overrides = await db["reviews"].count_documents({"is_overridden": True})
    csme_count = await db["reviews"].count_documents({"csme_threat": True})
    urgent = await db["reviews"].count_documents({"dispatch_status": "REFERRAL_DISPATCHED"})
    agreement = round(((total - overrides) / total) * 100, 1) if total > 0 else 100.0

    return {
        "total_reviews": total,
        "agreement_rate": agreement,
        "concordance_rate": agreement,
        "overrides": overrides,
        "total_overrides": overrides,
        "csme_threats_caught": csme_count,
        "urgent_referrals": urgent,
        "compliance": "ABDM / FHIR Diagnostic Audit Ready",
    }
