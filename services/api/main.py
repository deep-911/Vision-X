import sys
from pathlib import Path

# Add project root to sys.path to support running directly or as a module
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

if __package__ is None or __package__ == "":
    __package__ = "services.api"

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .database import connect_to_mongo, close_mongo_connection
from .routers import auth_router, cases_router, phc_router, reviews_router, fhir_router, screening_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visionx.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Connect to MongoDB and initialize indices/seed data
    logger.info("Initializing VisionX FastAPI application...")
    await connect_to_mongo()
    yield
    # Shutdown: Close DB connections cleanly
    logger.info("Shutting down VisionX FastAPI application...")
    await close_mongo_connection()


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="VisionX Retinal Screening API: Real-time diabetic retinopathy classification, lesion segmentation, and macular risk assessment.",
    version="2026.09.16",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# CORS middleware for modern frontend (Vite React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(auth_router.router, prefix=settings.API_V1_STR)
app.include_router(cases_router.router, prefix=settings.API_V1_STR)
app.include_router(screening_router.router, prefix=settings.API_V1_STR)
app.include_router(screening_router.router)
app.include_router(phc_router.router, prefix=settings.API_V1_STR)
app.include_router(reviews_router.router, prefix=settings.API_V1_STR)
app.include_router(fhir_router.router, prefix=settings.API_V1_STR)


@app.get("/api/v1/health", tags=["Health"])
@app.get("/health", tags=["Health"])
async def health_check():
    """System health and model readiness probe."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "api_version": "2026.09.16",
        "models": {
            "two_stage_pipeline_present": True,
            "lesion_unet_present": settings.LESION_UNET_PATH.is_file(),
            "architecture": "Two-Stage Referral Gatekeeper + Severity Stager (SIH26038)",
        },
        "database": f"SQLite ({settings.DATABASE_PATH.name})",
        "deployment": "Local Clinic / Cloud Screening Server",
    }


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "status": "online",
        "docs": f"{settings.API_V1_STR}/docs",
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("services.api.main:app", host="0.0.0.0", port=8000, reload=True)
