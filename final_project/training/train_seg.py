import os
import sys
import yaml
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset
from final_project.models.segformer import SegformerMultiLabel
from final_project.models.unet import get_unet_model
from final_project.training.losses import DiceBCELoss, WeightedFocalDiceLoss

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'train', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=config['data']['augment'],
        mode='train',
        sampling_strategy=config['data'].get('sampling_strategy', 'lesion_aware')
    )
    
    val_ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'val', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='val'
    )

    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
    # For validation sliding window, batch size must be 1 since image shapes are large (4288x2848)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config['dataset']['num_workers'])

    # Model
    if config['model']['architecture'] == 'SegFormer-B2':
        model = SegformerMultiLabel(num_classes=config['model']['num_classes']).to(device)
    elif config['model']['architecture'] == 'U-Net':
        model = get_unet_model(encoder_name=config['model'].get('encoder_name', 'efficientnet-b0'), num_classes=config['model']['num_classes']).to(device)
    else:
        raise ValueError(f"Unknown architecture {config['model']['architecture']}")
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    
    loss_cfg = config['training'].get('loss', {})
    if loss_cfg.get('name') == 'WeightedFocalDiceLoss':
        criterion = WeightedFocalDiceLoss(
            gamma=loss_cfg.get('gamma', 2.0),
            alpha=loss_cfg.get('alpha', 0.25),
            class_weights=loss_cfg.get('class_weights')
        )
    else:
        criterion = DiceBCELoss()
        
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

    # Dirs
    ckpt_dir = base_dir / "artifacts/checkpoints" / config['experiment_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_dice = -1.0

    # Training Loop
    epochs = config['training']['epochs']
    
    # We will use simple BCE loss on validation just for model selection checkpointing
    # The full evaluation (Dice, IoU) happens in eval_seg.py using sliding window.
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        for i, (imgs, masks, validities, names) in enumerate(train_loader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            validities = validities.to(device)
            
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                logits = model(imgs)
                loss = criterion(logits, masks, validities) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accumulation_steps
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_dice_sum = 0.0
        val_dice_count = 0
        
        with torch.no_grad():
            for imgs, masks, validities, names in val_loader:
                h, w = imgs.shape[2], imgs.shape[3]
                cy, cx = h//2, w//2
                ps = 512
                imgs_crop = imgs[:, :, cy-ps//2:cy+ps//2, cx-ps//2:cx+ps//2].to(device)
                masks_crop = masks[:, :, cy-ps//2:cy+ps//2, cx-ps//2:cx+ps//2].to(device)
                validities = validities.to(device)
                
                with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                    logits = model(imgs_crop)
                    loss = criterion(logits, masks_crop, validities)
                val_loss += loss.item()
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                # Compute macro dice for valid classes in this batch
                for b in range(preds.size(0)):
                    for c in range(preds.size(1)):
                        if validities[b, c] > 0:
                            p = preds[b, c]
                            g = masks_crop[b, c]
                            inter = (p * g).sum()
                            dice = (2.0 * inter) / (p.sum() + g.sum() + 1e-6)
                            val_dice_sum += dice.item()
                            val_dice_count += 1
                
        val_loss /= len(val_loader)
        epoch_macro_dice = val_dice_sum / max(val_dice_count, 1)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val CenterCrop Loss: {val_loss:.4f} - Val Macro Dice: {epoch_macro_dice:.4f}")
        
        # Save latest
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        
        if epoch_macro_dice > best_val_dice:
            best_val_dice = epoch_macro_dice
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print("  --> Saved new best checkpoint (Dice)!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
