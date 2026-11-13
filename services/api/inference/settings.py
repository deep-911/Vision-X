"""Central configuration for the VisionX inference engine."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = PROJECT_ROOT / "data" / "demo_samples"
MODELS_DIR = PROJECT_ROOT / "models"

STAGE1_GATE_PATH = MODELS_DIR / "production" / "stage1_gate.pt"
STAGE2_STAGER_PATH = MODELS_DIR / "production" / "stage2_stager.pt"
LESION_UNET_PATH = MODELS_DIR / "production" / "lesion_unet.pt"
TWO_STAGE_BACKBONE = "resnet18"
PREFER_FAST_ENGINE = True

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
INFERENCE_TIMEOUT_SECONDS = 90

# Versioning & Loopback Binding
API_VERSION = "2026.09.16"
BUILD_REVISION = "2026-09-16-sih26038-production"
HOST = "127.0.0.1"
PORT = 8000

REFERRAL_DISPATCH_ENABLED = True
