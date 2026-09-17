import os
import zipfile
import json
import hashlib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "Datasets"
IDRID_ZIP_DIR = DATASETS_DIR / "IDRiD"
IDRID_EXTRACT_DIR = DATASETS_DIR / "IDRiD_extracted"

def extract_and_manifest():
    print("Extracting IDRiD and generating manifest...")
    IDRID_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_files = ["A. Segmentation.zip", "B. Disease Grading.zip", "C. Localization.zip"]
    
    manifest = {
        "dataset_name": "IDRiD",
        "status": "extracted",
        "sub_datasets": {}
    }
    
    for zf_name in zip_files:
        zf_path = IDRID_ZIP_DIR / zf_name
        if not zf_path.exists():
            print(f"Warning: {zf_name} not found.")
            continue
            
        print(f"Extracting {zf_name}...")
        extract_sub_dir = IDRID_EXTRACT_DIR / zf_name.replace(".zip", "")
        extract_sub_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zf_path, 'r') as zip_ref:
                zip_ref.extractall(extract_sub_dir)
            
            manifest["sub_datasets"][zf_name] = {
                "archive": str(zf_path.relative_to(BASE_DIR)),
                "extracted_path": str(extract_sub_dir.relative_to(BASE_DIR)),
                "files_extracted": len([f for f in extract_sub_dir.rglob("*") if f.is_file()])
            }
        except zipfile.BadZipFile:
            print(f"Error: {zf_name} is not a valid zip file.")

    manifest_path = IDRID_EXTRACT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    extract_and_manifest()
