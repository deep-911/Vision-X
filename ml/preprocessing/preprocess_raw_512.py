import os
import cv2
import time
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "aptos2019"
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "aptos2019" / "train.csv"
OUTPUT_DIR_512 = PROJECT_ROOT / "data" / "processed" / "aptos_true_512"

CLASS_MAP = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

def crop_and_enhance(path_str, target_size=512):
    img = cv2.imread(path_str)
    if img is None:
        return None
    
    # 1. Detect retinal bounding box (remove dead black borders)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 10
    if mask.any():
        coords = cv2.findNonZero(mask.astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]
            
    # 2. Resize directly with area interpolation for anti-aliasing
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # 3. Ben Graham's local color constancy enhancement
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=target_size / 30)
    enhanced = cv2.addWeighted(img, 4, blurred, -4, 128)
    
    return enhanced

def process_single_image(item):
    id_code, diagnosis, src_path = item
    class_name = CLASS_MAP[diagnosis]
    out_path = OUTPUT_DIR_512 / class_name / f"{id_code}.png"
    
    if out_path.exists() and out_path.stat().st_size > 1000:
        return True
        
    enhanced = crop_and_enhance(str(src_path), target_size=512)
    if enhanced is not None:
        cv2.imwrite(str(out_path), enhanced)
        return True
    return False

def main():
    print("===================================================================")
    print(" VisionX: High-Resolution True 512x512 Dataset Preprocessor")
    print(" Sourcing from Raw 3,216x2,136 Uncompressed Fundus Scans")
    print("===================================================================")
    
    t0 = time.time()
    
    # Create target folders
    for class_name in CLASS_MAP.values():
        (OUTPUT_DIR_512 / class_name).mkdir(parents=True, exist_ok=True)
        
    df = pd.read_csv(CSV_PATH)
    print(f"[DATA] Loaded manifest: {len(df)} images.")
    
    # Index all raw image files
    print("[SCAN] Indexing raw high-resolution scan locations...")
    raw_files = {}
    for ext in ("*.png", "*.jpg"):
        for f in RAW_DIR.rglob(ext):
            raw_files[f.stem] = f
            
    print(f"[SCAN] Indexed {len(raw_files)} raw scan paths.")
    
    work_items = []
    missing = 0
    for _, row in df.iterrows():
        id_code = str(row["id_code"])
        diag = int(row["diagnosis"])
        if id_code in raw_files:
            work_items.append((id_code, diag, raw_files[id_code]))
        else:
            missing += 1
            
    print(f"[WORK] Prepared {len(work_items)} scans for processing. Missing: {missing}")
    
    # Multi-threaded execution
    print("[EXEC] Starting multi-threaded cropping & Ben Graham enhancement (12 workers)...")
    processed = 0
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(process_single_image, item) for item in work_items]
        for f in as_completed(futures):
            if f.result():
                processed += 1
            if processed % 500 == 0 or processed == len(work_items):
                elapsed = time.time() - t0
                rate = processed / max(elapsed, 0.1)
                print(f"  --> Processed {processed}/{len(work_items)} ({processed/len(work_items)*100:.1f}%) at {rate:.1f} scans/sec")
                
    total_time = time.time() - t0
    print(f"\n[DONE] Preprocessed {processed} high-resolution scans in {total_time:.1f}s ({total_time/60:.2f} min).")
    
    # Verification
    counts = {}
    for class_name in CLASS_MAP.values():
        c = len(list((OUTPUT_DIR_512 / class_name).glob("*.png")))
        counts[class_name] = c
    print("[VERIFY] Generated true 512x512 class distribution:")
    for k, v in counts.items():
        print(f"  • {k:16s}: {v} scans")
    print(f"Total True 512x512 Images: {sum(counts.values())}")

if __name__ == "__main__":
    main()
