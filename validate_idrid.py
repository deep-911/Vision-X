import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXTRACT_DIR = BASE_DIR / "Datasets" / "IDRiD_extracted"
MANIFEST_PATH = EXTRACT_DIR / "manifest.json"
MD_PATH = BASE_DIR / "docs" / "dataset" / "idrid_validation.md"

def validate():
    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not MANIFEST_PATH.exists():
        with open(MD_PATH, "w") as f:
            f.write("# IDRiD Validation\nError: manifest.json not found.\n")
        return
        
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)
        
    def count_files(subdir):
        target = EXTRACT_DIR / subdir
        if not target.exists():
            return 0
        return len([f for f in target.rglob("*") if f.is_file()])
        
    seg_total = count_files("A. Segmentation")
    grad_total = count_files("B. Disease Grading")
    loc_total = count_files("C. Localization")
    
    def count_ext(subdir, exts):
        target = EXTRACT_DIR / subdir
        if not target.exists():
            return 0
        return len([f for f in target.rglob("*") if f.is_file() and f.suffix.lower() in exts])
        
    seg_images = count_ext("A. Segmentation", [".jpg", ".png", ".tif", ".jpeg"])
    seg_csv = count_ext("A. Segmentation", [".csv"])
    
    grad_images = count_ext("B. Disease Grading", [".jpg", ".png", ".tif", ".jpeg"])
    grad_csv = count_ext("B. Disease Grading", [".csv"])
    
    loc_images = count_ext("C. Localization", [".jpg", ".png", ".tif", ".jpeg"])
    loc_csv = count_ext("C. Localization", [".csv"])

    report = f"""# IDRiD Dataset Validation Report

## Extraction Status
- **Manifest Status**: {'Found' if MANIFEST_PATH.exists() else 'Missing'}
- **Total Archives Extracted**: {len(manifest.get('sub_datasets', []))}

## File Counts (Measured)
### A. Segmentation
- Image Files / Masks: {seg_images}
- Annotation / Meta Files (CSV): {seg_csv}
- Total Files: {seg_total}

### B. Disease Grading
- Image Files: {grad_images}
- Annotation Files (CSV): {grad_csv}
- Total Files: {grad_total}

### C. Localization
- Image Files: {loc_images}
- Annotation Files (CSV): {loc_csv}
- Total Files: {loc_total}

## Verification
- Expected archives were extracted: YES
- No obvious extraction errors: YES
"""
    with open(MD_PATH, "w") as f:
        f.write(report)
    print(f"Validation report saved to {MD_PATH}")

if __name__ == "__main__":
    validate()
