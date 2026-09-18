import os
import sys
import yaml
import numpy as np
import cv2
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset

def main():
    print("Performing MA Component Audit...")
    
    config_path = base_dir / "final_project/configs/idrid_unet_effb0.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'train', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='val' # Load full images
    )
    
    total_components = 0
    images_with_ma = 0
    areas = []
    widths = []
    heights = []
    components_per_image = []
    
    for i in tqdm(range(len(ds.img_names))):
        name = ds.img_names[i]
        img_path = os.path.join(ds.img_dir, name + '.jpg')
        img = np.array(Image.open(img_path))
        h, w = img.shape[:2]
        
        mask, validity = ds._load_mask(name, (h, w))
        
        # MA is channel 0
        ma_mask = mask[:, :, 0].astype(np.uint8)
        
        if validity[0] > 0 and ma_mask.sum() > 0:
            images_with_ma += 1
            # Connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(ma_mask, connectivity=8)
            
            # stats: [x, y, w, h, area]
            # label 0 is background
            num_ma = num_labels - 1
            components_per_image.append(num_ma)
            total_components += num_ma
            
            for j in range(1, num_labels):
                area = stats[j, cv2.CC_STAT_AREA]
                width = stats[j, cv2.CC_STAT_WIDTH]
                height = stats[j, cv2.CC_STAT_HEIGHT]
                
                areas.append(area)
                widths.append(width)
                heights.append(height)
        else:
            components_per_image.append(0)

    print("\n=== MA Connected Component Audit ===")
    print(f"Total training images: {len(ds.img_names)}")
    print(f"Images containing MA: {images_with_ma}")
    print(f"Total MA components: {total_components}")
    print(f"Average components per MA-positive image: {np.mean([c for c in components_per_image if c > 0]):.2f}")
    print(f"Component Area (pixels): Mean {np.mean(areas):.1f} | Min {np.min(areas)} | Max {np.max(areas)} | Median {np.median(areas)}")
    print(f"Component Bounding Box Width: Mean {np.mean(widths):.1f} | Min {np.min(widths)} | Max {np.max(widths)}")
    print(f"Component Bounding Box Height: Mean {np.mean(heights):.1f} | Min {np.min(heights)} | Max {np.max(heights)}")

if __name__ == "__main__":
    main()
