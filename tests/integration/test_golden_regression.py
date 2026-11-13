"""Golden Benchmark Regression Assertion Test for VisionX."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.api.inference_service import get_benchmark_image, run_full_pipeline

def test_golden_grade2_regression():
    case_id = "000c1434d8d7"
    img = get_benchmark_image(case_id)
    assert img is not None, f"Failed to load benchmark scan {case_id}"

    result = run_full_pipeline(img, {"id": case_id, "patient_name": "Anand Verma"})

    assert result["grade"] == 2
    assert result["csme_threat"] is True
    assert result["min_fovea_dist_dd"] == 0.25
    assert result["is_referable"] is True
    assert result["analysis_status"] == "MODEL_REVIEW_REQUIRED"
    assert result["ex_count"] == 251
    assert result["he_count"] == 281

if __name__ == "__main__":
    test_golden_grade2_regression()
