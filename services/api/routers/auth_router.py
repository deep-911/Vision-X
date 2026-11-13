"""Authentication Router for VisionX."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import create_access_token, get_current_user, get_password_hash, verify_password
from ..database import get_db
from ..schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    user = await db["users"].find_one({"email": credentials.email})
    if not user or not verify_password(credentials.password, user.get("hashed_password", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(
        data={"sub": user["email"], "role": user["role"], "name": user["name"]}
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        name=user["name"],
        email=user["email"],
    )


@router.get("/me", response_model=UserOut)
async def get_current_user_profile(user: dict = Depends(get_current_user)):
    return UserOut(
        email=user["email"],
        name=user["name"],
        role=user["role"],
        center=user.get("center"),
        created_at=user.get("created_at"),
    )


@router.post("/register", response_model=UserOut)
async def register_user(req: RegisterRequest):
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    existing = await db["users"].find_one({"email": req.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")

    new_user = {
        "email": req.email,
        "name": req.name,
        "role": req.role,
        "center": req.center,
        "hashed_password": get_password_hash(req.password),
        "created_at": datetime.now(timezone.utc),
    }
    await db["users"].insert_one(new_user)
    return UserOut(
        email=new_user["email"],
        name=new_user["name"],
        role=new_user["role"],
        center=new_user["center"],
        created_at=new_user["created_at"],
    )


@router.get("/demo-users")
async def list_demo_users():
    """Provides quick demo user switching."""
    return [
        {
            "email": "doctor@visionx.health",
            "name": "Dr. Ramesh Gupta, MS",
            "role": "specialist",
            "title": "Specialist Ophthalmologist",
            "center": "District Hospital Eye Center",
            "default_password": "Password123!",
        },
        {
            "email": "worker@visionx.health",
            "name": "Priya Sharma, ANM",
            "role": "phc_worker",
            "title": "Rural PHC Health Worker",
            "center": "Primary Health Centre (PHC-04)",
            "default_password": "Password123!",
        },
        {
            "email": "patient@visionx.health",
            "name": "Anand Verma",
            "role": "patient",
            "title": "Screened Diabetic Patient",
            "center": "PHC Baramati",
            "default_password": "Password123!",
        },
    ]


@router.post("/quick-switch/{role}", response_model=TokenResponse)
async def quick_switch_persona(role: str):
    """Switch user role and return access token for specialist, clinic worker, or patient."""
    db = get_db()
    if db is None:
        raise HTTPException(status_code=500, detail="Database unavailable")

    user = await db["users"].find_one({"role": role})
    if not user:
        # Fallback query by email if role not found directly
        email_map = {
            "specialist": "doctor@visionx.health",
            "phc_worker": "worker@visionx.health",
            "patient": "patient@visionx.health",
        }
        if role in email_map:
            user = await db["users"].find_one({"email": email_map[role]})

    if not user:
        raise HTTPException(status_code=404, detail=f"User with role '{role}' not found.")

    token = create_access_token(
        data={"sub": user["email"], "role": user["role"], "name": user["name"]}
    )
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        role=user["role"],
        name=user["name"],
        email=user["email"],
    )
