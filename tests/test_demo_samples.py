import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from services.api.inference_service import run_full_pipeline

def test_demo_samples():
    demo_dir = Path("data/demo_samples")
    demo_files = sorted(list(demo_dir.glob("*.jpg")))
    print(f"Found {len(demo_files)} demo samples in {demo_dir}:")
    assert len(demo_files) > 0, "No demo samples found!"

    results = []
    for f in demo_files:
        img_bgr = cv2.imread(str(f))
        assert img_bgr is not None, f"Failed to read {f}"
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        meta = {"id": f.stem, "patient_name": f"Demo Patient ({f.stem})"}
        res = run_full_pipeline(img_rgb, meta)
        results.append(res)
        print(f"\n==========================================")
        print(f"Sample: {f.name} (Shape: {img_rgb.shape})")
        print(f"Quality Status: {res['quality']['status']} | Composite Quality Index: {res['cqi']:.2f}")
        print(f"Analysis Status: {res['analysis_status']}")
        print(f"Predicted Grade: {res.get('grade')} ({res.get('grade_name')})")
        print(f"Confidence: {res.get('confidence')}% | Referable Risk: {res.get('p_referable')}%")
        print(f"CSME Threat: {res.get('csme_threat')} | Fovea Dist: {res.get('min_fovea_dist_dd')} DD")
        print(f"Detected Lesions: EX={res.get('ex_count')} px, HE={res.get('he_count')} px, SE={res.get('se_count')} px")
        print(f"Lesion Model: {res.get('lesion_model_name')} (Connected: {res.get('lesion_model_connected')})")
        print(f"Images Produced: Pristine={bool(res['images']['pristine'])}, CLAHE={bool(res['images']['clahe'])}, Lesions={bool(res['images']['lesions'])}, GradCAM={bool(res['images']['gradcam'])}")
        assert res["images"]["pristine"].startswith("data:image/jpeg;base64,")
        assert res["images"]["clahe"].startswith("data:image/jpeg;base64,")
        assert res["images"]["lesions"].startswith("data:image/jpeg;base64,")

    print("\n------------------------------------------")
    print(f"All {len(demo_files)} demo samples successfully processed by inference engine!")
    assert len(results) == len(demo_files)
    required_keys = [
        "analysis_status",
        "patient_meta",
        "quality",
        "cqi",
        "images",
        "total_lesions",
        "csme_threat",
        "grade",
        "grade_name",
        "confidence",
        "p_referable",
    ]
    for r in results:
        for k in required_keys:
            assert k in r, f"Missing key '{k}' in result"
        assert r["analysis_status"] in (
            "ANALYSIS_COMPLETE",
            "MODEL_REVIEW_REQUIRED",
            "MODEL_REVIEW_REQUIRED_ESCALATED",
            "QUALITY_REJECTED",
        )

if __name__ == "__main__":
    test_demo_samples()
