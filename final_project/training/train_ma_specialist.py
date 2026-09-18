import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from final_project.data.idrid_ma_specialist import IDRiDMASpecialistDataset
from final_project.models.unet import get_unet_model
from final_project.training.losses import DiceBCELoss

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    torch.manual_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Checkpoints dir
    ckpt_dir = base_dir / "artifacts/checkpoints" / config['experiment_name']
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    train_ds = IDRiDMASpecialistDataset(
        config['data']['dataset_root'], 
        'train', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=config['data']['augment'],
        mode='train'
    )
    
    val_ds = IDRiDMASpecialistDataset(
        config['data']['dataset_root'], 
        'val', 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='val'
    )
    
    train_loader = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], shuffle=True, num_workers=config['dataset']['num_workers'])
    # Validation loop using center crops instead of full sliding window for speed during training
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=config['dataset']['num_workers'])
    
    # Model
    model = get_unet_model(encoder_name=config['model'].get('encoder_name', 'efficientnet-b0'), num_classes=1).to(device)
    
    # Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'])
    criterion = DiceBCELoss()
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    accumulation_steps = config['training'].get('gradient_accumulation_steps', 1)
    
    best_val_dice = -1
    
    for epoch in range(1, config['training']['epochs'] + 1):
        model.train()
        train_loss = 0.0
        
        optimizer.zero_grad()
        for i, (imgs, masks, validities, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{config['training']['epochs']} Train")):
            imgs = imgs.to(device)
            masks = masks.to(device)
            validities = validities.to(device)
            
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                logits = model(imgs)
                loss = criterion(logits, masks, validities)
                loss = loss / accumulation_steps
                
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
            train_loss += loss.item() * accumulation_steps
            
        train_loss /= len(train_loader)
        
        # Quick validation using a single center crop for the epoch monitor
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for imgs, masks, validities, _ in val_loader:
                # imgs: [1, 3, 2848, 4288] -> Crop center 512x512
                # Since we just want a quick directional metric, we take center crop
                h, w = imgs.shape[2], imgs.shape[3]
                cy, cx = h // 2, w // 2
                ps = 512
                
                img_crop = imgs[:, :, cy-ps//2:cy+ps//2, cx-ps//2:cx+ps//2].to(device)
                mask_crop = masks[:, :, cy-ps//2:cy+ps//2, cx-ps//2:cx+ps//2].to(device)
                val_crop = validities.to(device)
                
                with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                    logits = model(img_crop)
                    v_loss = criterion(logits, mask_crop, val_crop)
                    
                val_loss += v_loss.item()
                
                # Compute Dice quickly
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                
                intersection = (preds * mask_crop).sum()
                dice = (2. * intersection) / (preds.sum() + mask_crop.sum() + 1e-6)
                val_dice += dice.item()
                val_batches += 1
                
        val_loss /= val_batches
        val_dice /= val_batches
        
        print(f"Epoch {epoch}/{config['training']['epochs']} - Train Loss: {train_loss:.4f} - Val CenterCrop Loss: {val_loss:.4f} - Val CenterCrop MA Dice: {val_dice:.4f}")
        
        if val_dice >= best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print("  --> Saved new best checkpoint (Dice)!")
            
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
