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
from final_project.training.losses import DiceBCELoss

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
        mode='train'
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
    model = SegformerMultiLabel(num_classes=config['model']['num_classes']).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)

    # Dirs
    ckpt_dir = base_dir / "artifacts/checkpoints" / config['experiment_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')

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
        
        with torch.no_grad():
            for imgs, masks, validities, names in val_loader:
                # Valid sliding window emulation: just evaluating loss on random 512 patches for speed
                # Actually, eval_seg.py does full reconstruction. Here we just take a center crop to track val loss cheaply.
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
                
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val CenterCrop Loss: {val_loss:.4f}")
        
        # Save latest
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print("  --> Saved new best checkpoint!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
