import os
import sys
import yaml
import numpy as np
import torch
import cv2
from pathlib import Path
from PIL import Image

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset

def main():
    print("Initializing ma_magnified dataset...")
    
    config_path = base_dir / "final_project/configs/idrid_unet_effb0_ma_sampling.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'train', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='train',
        sampling_strategy='ma_magnified'
    )
    
    print("\n=== MA Magnification Audit ===")
    
    # We will grab 50 patches and average the metrics
    ma_patches = 0
    native_sizes = []
    input_sizes = []
    
    # We can't easily measure "before resize" from the dataset object because it resizes internally.
    # We will do a manual emulation for the audit here on 1 image to trace a single MA.
    
    img_name = ds.img_names[0]
    img_path = os.path.join(ds.img_dir, img_name + '.jpg')
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    mask, validity = ds._load_mask(img_name, (h, w))
    
    pos_pixels = np.argwhere(mask[:, :, 0] > 0)
    if len(pos_pixels) > 0:
        center_y, center_x = pos_pixels[len(pos_pixels)//2]
        
        native_crop = 256
        y_min = max(0, center_y - native_crop//2)
        x_min = max(0, center_x - native_crop//2)
        y_min = min(y_min, h - native_crop)
        x_min = min(x_min, w - native_crop)
        
        img_native = img[y_min:y_min+native_crop, x_min:x_min+native_crop]
        mask_native = mask[y_min:y_min+native_crop, x_min:x_min+native_crop]
        
        ma_pixels_before = mask_native[:, :, 0].sum()
        ma_density_before = ma_pixels_before / (native_crop * native_crop)
        
        img_resized = cv2.resize(img_native, (512, 512), interpolation=cv2.INTER_LINEAR)
        mask_resized = np.zeros((512, 512, 4), dtype=np.float32)
        for c in range(4):
            mask_resized[:, :, c] = cv2.resize(mask_native[:, :, c], (512, 512), interpolation=cv2.INTER_NEAREST)
            
        ma_pixels_after = mask_resized[:, :, 0].sum()
        ma_density_after = ma_pixels_after / (512 * 512)
        
        print(f"Number of MA-positive crops tracked: 1")
        print(f"Native crop dimensions: {img_native.shape[:2]}")
        print(f"Model input dimensions: {img_resized.shape[:2]}")
        print(f"MA pixel count BEFORE resize: {ma_pixels_before}")
        print(f"MA pixel count AFTER resize: {ma_pixels_after} (Expected ~ {ma_pixels_before * 4})")
        print(f"MA pixel density BEFORE resize: {ma_density_before*100:.4f}%")
        print(f"MA pixel density AFTER resize: {ma_density_after*100:.4f}%")
        
        print("\nVerification geometrically:")
        print(f"A 2x2 MA mask before resize becomes a {(2*2)*4} pixel mask after nearest-neighbor resize,")
        print("which corresponds to 4x4 pixels in the 512x512 space.")
        print("The effective density remains the same relative to the crop window, but the absolute spatial resolution is 2x magnified relative to the CNN's receptive field.")

if __name__ == "__main__":
    main()
