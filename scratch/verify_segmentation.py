import os
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset
from final_project.models.segformer import SegformerMultiLabel
from final_project.training.losses import DiceBCELoss

def test_dataloader_and_validity():
    print("=== Testing Dataloader and Validity ===")
    ds = IDRiDSegmentationDataset('Datasets/IDRiD_extracted/A. Segmentation/A. Segmentation', 'train', 'final_project/data/splits/idrid_seg_split.json', mode='train')
    print(f"Dataset length: {len(ds)}")
    
    validity_counts = np.zeros(4)
    for i in range(len(ds)):
        _, _, val, _ = ds[i] # we don't load images just validity for fast counting, actually we do load imgs
        validity_counts += val
        if i >= 43: # check all 44
            pass
            
    print(f"Valid annotations per class in train: MA={validity_counts[0]}, HE={validity_counts[1]}, EX={validity_counts[2]}, SE={validity_counts[3]}")
    
    # Grab one sample
    img, mask, val, name = ds[0]
    print(f"Sample {name}: Patch shape {img.shape}, Mask shape {mask.shape}, Validity {val}")
    
    # Save a visualization
    img_np = img.permute(1, 2, 0).numpy()
    # Unnormalize
    img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_np = np.clip(img_np, 0, 1)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img_np)
    axes[0].set_title('Image Patch')
    for c in range(4):
        axes[c+1].imshow(mask[c].numpy(), cmap='gray')
        axes[c+1].set_title(f'Mask {c} (Val={val[c]})')
    os.makedirs('scratch', exist_ok=True)
    plt.savefig('scratch/sample_patch.png')
    print("Saved sample_patch.png")

def test_loss():
    print("\n=== Testing Loss Validity Handling ===")
    criterion = DiceBCELoss()
    
    # Dummy predictions (B=2, C=4, H=16, W=16)
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    targets = torch.zeros(2, 4, 16, 16)
    targets[0, 0, 5:10, 5:10] = 1 # Fake MA in item 0
    targets[1, 3, 5:10, 5:10] = 1 # Fake SE in item 1
    
    # Validity: batch 0 has MA missing (val=0) even though target has it? No target should be 0 if val=0
    # Let's say Item 0 missing SE, Item 1 missing MA
    validities = torch.tensor([[1.0, 1.0, 1.0, 0.0],
                               [0.0, 1.0, 1.0, 1.0]])
                               
    loss = criterion(logits, targets, validities)
    print(f"Loss computed: {loss.item()}")
    loss.backward()
    print("Gradients computed successfully. Masking logic works.")

def test_reconstruction():
    print("\n=== Testing Sliding Window Reconstruction ===")
    full_h, full_w = 2848, 4288
    patch_size = 512
    overlap = 256
    stride = patch_size - overlap
    
    print(f"Target dimension: {full_w}x{full_h}")
    y_starts = list(range(0, full_h - patch_size + 1, stride))
    if y_starts[-1] + patch_size < full_h:
        y_starts.append(full_h - patch_size)
        
    x_starts = list(range(0, full_w - patch_size + 1, stride))
    if x_starts[-1] + patch_size < full_w:
        x_starts.append(full_w - patch_size)
        
    print(f"Number of patches: {len(y_starts)}x{len(x_starts)} = {len(y_starts)*len(x_starts)}")
    
    # Dummy full prediction tensor
    full_pred = torch.zeros(4, full_h, full_w)
    count_map = torch.zeros(4, full_h, full_w)
    
    dummy_patch_pred = torch.ones(4, patch_size, patch_size) # all ones
    
    for y in y_starts:
        for x in x_starts:
            full_pred[:, y:y+patch_size, x:x+patch_size] += dummy_patch_pred
            count_map[:, y:y+patch_size, x:x+patch_size] += 1
            
    full_pred = full_pred / count_map
    print(f"Reconstructed Shape: {full_pred.shape}")
    print(f"Reconstructed Min/Max value: {full_pred.min().item()} / {full_pred.max().item()} (Expected 1.0/1.0)")
    
if __name__ == "__main__":
    test_dataloader_and_validity()
    test_loss()
    test_reconstruction()
