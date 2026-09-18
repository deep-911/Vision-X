import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import torch.optim as optim

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset
from final_project.models.segformer import SegformerMultiLabel
from final_project.training.losses import DiceBCELoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = base_dir / "scratch" / "debug_outputs"
out_dir.mkdir(parents=True, exist_ok=True)

# Helper function
def p(text):
    print(text)
    with open(out_dir / "debug_report.txt", "a") as f:
        f.write(text + "\n")

if os.path.exists(out_dir / "debug_report.txt"):
    os.remove(out_dir / "debug_report.txt")

p("=== PHASE 3A SEGFORMER DEBUG DIAGNOSTICS ===")

def audit_mask_data():
    p("\n--- 1. MASK / DATA AUDIT ---")
    ds = IDRiDSegmentationDataset('Datasets/IDRiD_extracted/A. Segmentation/A. Segmentation', 'train', 'final_project/data/splits/idrid_seg_split.json', mode='val')
    
    valid_counts = np.zeros(4)
    positive_img_counts = np.zeros(4)
    total_pos_pixels = np.zeros(4)
    total_valid_pixels = np.zeros(4)
    
    mask_min = float('inf')
    mask_max = float('-inf')
    unique_vals = set()
    img_shapes = set()
    mask_shapes = set()
    
    for i in range(len(ds)):
        img, mask, val, name = ds[i]
        
        c, h, w = img.shape
        img_shapes.add((h, w))
        mask_shapes.add((mask.shape[1], mask.shape[2]))
        
        valid_counts += val
        
        m_np = mask.numpy()
        mask_min = min(mask_min, m_np.min())
        mask_max = max(mask_max, m_np.max())
        unique_vals.update(np.unique(m_np).tolist())
        
        for c in range(4):
            if val[c] > 0:
                pos_pix = (m_np[c] > 0).sum()
                total_pos_pixels[c] += pos_pix
                total_valid_pixels[c] += (h * w)
                if pos_pix > 0:
                    positive_img_counts[c] += 1
                    
        # Visualize one full image and mask
        if i == 0:
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            img_np = img.permute(1,2,0).numpy()
            img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            axes[0].imshow(img_np)
            axes[0].set_title(f'Original {h}x{w}')
            for c in range(4):
                axes[c+1].imshow(m_np[c], cmap='gray')
                axes[c+1].set_title(f'Class {c} (Val={val[c]})')
            plt.savefig(out_dir / "audit_full_image.png")
            plt.close(fig)
            
    p(f"Image dimensions found: {img_shapes}")
    p(f"Mask dimensions found: {mask_shapes}")
    p(f"Mask min: {mask_min}, max: {mask_max}, unique values: {unique_vals}")
    p(f"Valid annotation count (MA, HE, EX, SE): {valid_counts}")
    p(f"Images with positive pixels (MA, HE, EX, SE): {positive_img_counts}")
    p(f"Total positive pixels: {total_pos_pixels}")
    p(f"Positive pixel percentage: {(total_pos_pixels / total_valid_pixels) * 100}%")

def audit_data_sampling():
    p("\n--- 2. DATA SAMPLING AUDIT ---")
    # Use train mode for patches
    ds = IDRiDSegmentationDataset('Datasets/IDRiD_extracted/A. Segmentation/A. Segmentation', 'train', 'final_project/data/splits/idrid_seg_split.json', mode='train')
    
    has_ma, has_ex, has_he, has_se = 0, 0, 0, 0
    has_any, has_none = 0, 0
    
    total = 200
    for i in range(total):
        # sample randomly
        img, mask, val, name = ds[np.random.randint(len(ds))]
        m_np = mask.numpy()
        
        ma_p = (m_np[0] > 0).sum() > 0
        he_p = (m_np[1] > 0).sum() > 0
        ex_p = (m_np[2] > 0).sum() > 0
        se_p = (m_np[3] > 0).sum() > 0
        
        if ma_p: has_ma += 1
        if he_p: has_he += 1
        if ex_p: has_ex += 1
        if se_p: has_se += 1
        
        if any([ma_p, he_p, ex_p, se_p]):
            has_any += 1
        else:
            has_none += 1
            
        # Visualize one cropped patch
        if i == 0:
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            img_np = img.permute(1,2,0).numpy()
            img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
            axes[0].imshow(img_np)
            axes[0].set_title(f'Crop {img.shape[1:]}')
            for c in range(4):
                axes[c+1].imshow(m_np[c], cmap='gray')
                axes[c+1].set_title(f'Crop Mask {c} (Val={val[c]})')
            plt.savefig(out_dir / "audit_cropped_patch.png")
            plt.close(fig)

    p(f"Out of {total} random train patches:")
    p(f"MA presence: {has_ma / total:.1%}")
    p(f"HE presence: {has_he / total:.1%}")
    p(f"EX presence: {has_ex / total:.1%}")
    p(f"SE presence: {has_se / total:.1%}")
    p(f"Any lesion presence: {has_any / total:.1%}")
    p(f"No lesion presence: {has_none / total:.1%}")

def audit_model_and_normalization():
    p("\n--- 3. PRETRAINED MODEL / NORMALIZATION AUDIT ---")
    model = SegformerMultiLabel(num_classes=4).to(device)
    
    # 1 batch
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    out = model(dummy_input)
    
    p(f"Input shape: {dummy_input.shape}")
    p(f"Input min/max/mean/std: {dummy_input.min().item():.3f} / {dummy_input.max().item():.3f} / {dummy_input.mean().item():.3f} / {dummy_input.std().item():.3f}")
    p(f"Output shape: {out.shape}")
    p(f"Output dtype: {out.dtype}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    p(f"Total Parameters: {total_params:,}")
    p(f"Trainable Parameters: {trainable_params:,}")
    
    # Check if backbone is trained
    p(f"Model backbone requires_grad: {list(model.model.segformer.parameters())[0].requires_grad}")
    
def audit_loss_and_overfit():
    p("\n--- 4. LOSS AUDIT & 6. CRITICAL OVERFIT TEST ---")
    
    import torch.nn.functional as F
    
    p("DiceBCELoss Formula:")
    p("  inputs_sigmoid = torch.sigmoid(inputs)")
    p("  inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)")
    p("  targets_flat = targets.view(...)")
    p("  bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none').mean(dim=2)")
    p("  intersection = (inputs_sigmoid_flat * targets_flat).sum(dim=2)")
    p("  dice = (2 * intersection + 1) / (inputs_sigmoid_flat.sum() + targets_flat.sum() + 1)")
    p("  dice_loss = 1 - dice")
    p("  total_loss = (bce_loss + dice_loss) * validities")
    p("  return total_loss.sum() / validities.sum()")
    
    criterion = DiceBCELoss()
    model = SegformerMultiLabel(num_classes=4).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda')
    
    # Select 2 images for overfit
    ds = IDRiDSegmentationDataset('Datasets/IDRiD_extracted/A. Segmentation/A. Segmentation', 'train', 'final_project/data/splits/idrid_seg_split.json', mode='val') # Use val mode to get full image, then we manually crop deterministic
    
    img1, mask1, val1, _ = ds[0]
    img2, mask2, val2, _ = ds[1]
    
    # Manually extract a deterministic 512x512 crop that definitely has lesions
    def extract_deterministic_crop(img, mask):
        pos = torch.argwhere(mask[0] > 0) # MA
        if len(pos) == 0:
            pos = torch.argwhere(mask > 0)[:, 1:] # Any
        
        y, x = pos[len(pos)//2]
        y_min = max(0, min(y - 256, img.shape[1] - 512))
        x_min = max(0, min(x - 256, img.shape[2] - 512))
        return img[:, y_min:y_min+512, x_min:x_min+512], mask[:, y_min:y_min+512, x_min:x_min+512]
        
    c_img1, c_mask1 = extract_deterministic_crop(img1, mask1)
    c_img2, c_mask2 = extract_deterministic_crop(img2, mask2)
    
    batch_imgs = torch.stack([c_img1, c_img2]).to(device)
    batch_masks = torch.stack([c_mask1, c_mask2]).to(device)
    batch_vals = torch.tensor(np.stack([val1, val2])).to(device)
    
    # Modify one validity to test exclusion (Force SE (idx 3) on img 1 to be 0)
    batch_vals[0, 3] = 0.0
    
    p(f"\nTraining on batch of 2 patches. Validities: \n{batch_vals.cpu().numpy()}")
    
    # Loss audit on first pass
    with torch.amp.autocast('cuda'):
        logits = model(batch_imgs)
        # break down loss
        inputs_flat = logits.view(logits.size(0), logits.size(1), -1)
        targets_flat = batch_masks.view(batch_masks.size(0), batch_masks.size(1), -1)
        bce = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none').mean(dim=2)
        
        inputs_sig = torch.sigmoid(logits).view(logits.size(0), logits.size(1), -1)
        inter = (inputs_sig * targets_flat).sum(dim=2)
        dice = (2*inter + 1)/(inputs_sig.sum(dim=2) + targets_flat.sum(dim=2) + 1)
        dice_l = 1 - dice
        
        p(f"Initial BCE per class (Img0): {bce[0].detach().cpu().numpy()}")
        p(f"Initial Dice per class (Img0): {dice_l[0].detach().cpu().numpy()}")
        
    p("\n--- 5. LOGIT/PROB DIAGNOSTIC (Before) ---")
    probs = torch.sigmoid(logits).detach()
    for c in range(4):
        p(f"Class {c}: Logit min={logits[:,c].min():.3f}, mean={logits[:,c].mean():.3f}, max={logits[:,c].max():.3f} | Prob mean={probs[:,c].mean():.3f}, max={probs[:,c].max():.3f}")
        p(f"  Pixels > 0.1: {(probs[:,c] > 0.1).float().mean():.2%} | > 0.25: {(probs[:,c] > 0.25).float().mean():.2%} | > 0.5: {(probs[:,c] > 0.5).float().mean():.2%}")
        
    p("\nStarting 100-step overfit test...")
    
    for step in range(100):
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            logits = model(batch_imgs)
            loss = criterion(logits, batch_masks, batch_vals)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if step % 20 == 0:
            p(f"Step {step} - Loss: {loss.item():.4f}")
            
    p("\n--- 5. LOGIT/PROB DIAGNOSTIC (After 100 steps) ---")
    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            logits = model(batch_imgs)
    probs = torch.sigmoid(logits).detach()
    for c in range(4):
        p(f"Class {c}: Logit min={logits[:,c].min():.3f}, mean={logits[:,c].mean():.3f}, max={logits[:,c].max():.3f} | Prob mean={probs[:,c].mean():.3f}, max={probs[:,c].max():.3f}")
        p(f"  Pixels > 0.1: {(probs[:,c] > 0.1).float().mean():.2%} | > 0.25: {(probs[:,c] > 0.25).float().mean():.2%} | > 0.5: {(probs[:,c] > 0.5).float().mean():.2%}")
    
    # Save after overfit
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    img_np = batch_imgs[0].cpu().permute(1,2,0).numpy()
    img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
    mask_np = batch_masks[0].cpu().numpy()
    pred_np = probs[0].cpu().numpy()
    
    for c in range(4):
        axes[c, 0].imshow(img_np)
        axes[c, 0].set_title(f'Original Crop')
        axes[c, 1].imshow(mask_np[c], cmap='gray')
        axes[c, 1].set_title(f'GT Class {c}')
        axes[c, 2].imshow(pred_np[c], cmap='viridis')
        axes[c, 2].set_title(f'Pred Prob Class {c}')
    plt.savefig(out_dir / "overfit_results.png")
    plt.close(fig)

if __name__ == "__main__":
    audit_mask_data()
    audit_data_sampling()
    audit_model_and_normalization()
    audit_loss_and_overfit()
