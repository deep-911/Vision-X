import os
import sys
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset
from final_project.models.unet import get_unet_model
from final_project.training.losses import DiceBCELoss

def main():
    config_path = base_dir / "final_project/configs/idrid_unet_effb0_ma_sampling.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'train', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='train',
        sampling_strategy='ma_focused'
    )
    
    # We want representative training batches. Let's run a few batches.
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
    
    # Load model
    model = get_unet_model(encoder_name=config['model']['encoder_name'], num_classes=config['model']['num_classes']).to(device)
    model.train() # Set to train so gradients flow properly
    
    classes = ['MA', 'HE', 'EX', 'SE']
    
    print("Running Pre-Training Loss Audit (10 batches)...")
    
    accum_bce = {c: [] for c in range(4)}
    accum_dice = {c: [] for c in range(4)}
    accum_grad = {c: [] for c in range(4)}
    
    for i, (imgs, masks, validities, names) in enumerate(loader):
        if i >= 10: break
        
        imgs = imgs.to(device)
        masks = masks.to(device)
        validities = validities.to(device)
        
        logits = model(imgs)
        
        for c in range(4):
            # Calculate loss separately for each channel
            bce = F.binary_cross_entropy_with_logits(logits[:, c], masks[:, c], reduction='none')
            probs = torch.sigmoid(logits[:, c])
            intersection = (probs * masks[:, c]).sum(dim=(1,2))
            dice = 1 - (2. * intersection + 1e-6) / (probs.sum(dim=(1,2)) + masks[:, c].sum(dim=(1,2)) + 1e-6)
            
            # Mask out invalid
            val_mask = validities[:, c]
            bce_loss = (bce.mean(dim=(1,2)) * val_mask).sum() / (val_mask.sum() + 1e-6)
            dice_loss = (dice * val_mask).sum() / (val_mask.sum() + 1e-6)
            
            total_c_loss = bce_loss + dice_loss
            
            accum_bce[c].append(bce_loss.item())
            accum_dice[c].append(dice_loss.item())
            
            if total_c_loss > 0:
                # Compute gradient with respect to logits
                model.zero_grad()
                total_c_loss.backward(retain_graph=True)
                
                # We can measure the gradient magnitude on the logits for this channel
                # Oh wait, logits gradient might not be retained unless we do logits.retain_grad().
                # Alternatively, measure gradient magnitude of the last conv layer weights!
                # smp Unet has segmentation_head.
                grad_mag = 0.0
                for name, param in model.segmentation_head.named_parameters():
                    if param.grad is not None:
                        # Extract gradient for the specific channel
                        # The segmentation head is usually a Conv2d(in, out_classes, 1)
                        if 'weight' in name:
                            # param shape: [4, in_channels, 1, 1]
                            # we can take the norm of the gradient for the c-th output channel
                            grad_mag += param.grad[c].norm().item()
                            
                accum_grad[c].append(grad_mag)

    print("\n--- LOSS CONTRIBUTION DIAGNOSTIC (Mean over 10 batches) ---")
    total_loss_all = sum(np.mean(accum_bce[c]) + np.mean(accum_dice[c]) for c in range(4))
    
    for c in range(4):
        mean_bce = np.mean(accum_bce[c])
        mean_dice = np.mean(accum_dice[c])
        total_class_loss = mean_bce + mean_dice
        rel_contrib = total_class_loss / (total_loss_all + 1e-6) * 100
        mean_grad = np.mean(accum_grad[c]) if len(accum_grad[c]) > 0 else 0
        
        print(f"Class: {classes[c]}")
        print(f"  BCE Contribution:  {mean_bce:.4f}")
        print(f"  Dice Contribution: {mean_dice:.4f}")
        print(f"  Total Class Loss:  {total_class_loss:.4f} ({rel_contrib:.1f}%)")
        print(f"  Grad Magnitude (Seg Head): {mean_grad:.6f}")

if __name__ == "__main__":
    main()
