"""Application Configuration for VisionX FastAPI Backend."""

import os
from pathlib import Path


class Settings:
    PROJECT_NAME: str = "VisionX Retinal Screening API"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "visionx-sih26038-telemedicine-super-secret-jwt-key-2026")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 12  # 12 hours

    # MongoDB connection (local running instance)
    MONGODB_URL: str = os.getenv("MONGODB_URL", "mongodb://127.0.0.1:27017")
    DATABASE_NAME: str = os.getenv("DATABASE_NAME", "visionx")

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    DATA_DIR: Path = PROJECT_ROOT / "data"

    # Canonical Database Path
    DATABASE_PATH: Path = DATA_DIR / "runtime" / "telemedicine.db"

    # Two-Stage Model Weights & Architecture
    STAGE1_GATE_PATH: Path = MODELS_DIR / "production" / "stage1_gate.pt"
    STAGE2_STAGER_PATH: Path = MODELS_DIR / "production" / "stage2_stager.pt"
    LESION_UNET_PATH: Path = MODELS_DIR / "production" / "lesion_unet.pt"
    TWO_STAGE_BACKBONE: str = "resnet18"

    @classmethod
    def resolve_model_path(cls, filename: str) -> Path:
        """Resolves model path looking in production then research."""
        prod = cls.MODELS_DIR / "production" / filename
        if prod.is_file():
            return prod
        research = cls.MODELS_DIR / "research" / filename
        if research.is_file():
            return research
        return cls.MODELS_DIR / filename

    # CORS Origins for Vite Frontend
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8050",
        "http://127.0.0.1:8050",
    ]


settings = Settings()
