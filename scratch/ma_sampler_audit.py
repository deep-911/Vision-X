import os
import sys
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset

def main():
    config_path = base_dir / "final_project/configs/idrid_unet_effb0.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Initializing dataset with ma_focused sampling...")
    ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'train', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='train',
        sampling_strategy='ma_focused'
    )
    
    print("Running Pre-Training Audit (500 patches)...")
    
    ma_count = 0
    he_count = 0
    ex_count = 0
    se_count = 0
    any_count = 0
    ma_positive_pixels_total = 0
    total_pixels = 0
    
    unique_ma_sources = set()
    
    for i in tqdm(range(500)):
        idx = np.random.randint(0, len(ds))
        img, mask, validity, base_name = ds[idx]
        
        # mask is [C, H, W]
        has_ma = (mask[0].sum() > 0).item()
        has_he = (mask[1].sum() > 0).item()
        has_ex = (mask[2].sum() > 0).item()
        has_se = (mask[3].sum() > 0).item()
        
        if has_ma: ma_count += 1
        if has_he: he_count += 1
        if has_ex: ex_count += 1
        if has_se: se_count += 1
        if has_ma or has_he or has_ex or has_se: any_count += 1
        
        ma_positive_pixels_total += mask[0].sum().item()
        total_pixels += mask[0].numel()
        
        if has_ma:
            unique_ma_sources.add(base_name)
            
    print(f"\n--- SAMPLING AUDIT RESULTS (500 patches) ---")
    print(f"% containing MA: {ma_count / 500 * 100:.1f}%")
    print(f"% containing HE: {he_count / 500 * 100:.1f}%")
    print(f"% containing EX: {ex_count / 500 * 100:.1f}%")
    print(f"% containing SE: {se_count / 500 * 100:.1f}%")
    print(f"% containing any lesion: {any_count / 500 * 100:.1f}%")
    print(f"MA positive pixel percentage within patches: {(ma_positive_pixels_total / total_pixels) * 100:.4f}%")
    print(f"Unique MA source images represented: {len(unique_ma_sources)}")

if __name__ == "__main__":
    main()
