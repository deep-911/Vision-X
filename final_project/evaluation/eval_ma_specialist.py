import os
import sys
import yaml
import json
import torch
import numpy as np
import cv2
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from final_project.data.idrid_ma_specialist import IDRiDMASpecialistDataset
from final_project.models.unet import get_unet_model
from final_project.evaluation.lesion_metrics import compute_lesion_metrics

def evaluate(config_path, checkpoint_path, split='val'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    ds = IDRiDMASpecialistDataset(
        config['data']['dataset_root'], 
        split, 
        config['data']['split_json'],
        patch_size=config['data']['patch_size'],
        augment=False,
        mode='val'
    )
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=config['dataset']['num_workers'])

    # Model
    model = get_unet_model(encoder_name=config['model'].get('encoder_name', 'efficientnet-b0'), num_classes=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    native_patch_size = 256
    model_input_size = 512
    overlap = native_patch_size // 2
    stride = native_patch_size - overlap
    
    # Thresholds for FROC and threshold sweep
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    metrics_per_thresh = {t: {
        'pixel_tp':0, 'pixel_fp':0, 'pixel_fn':0, 'pixel_tn':0, 
        'lesion_tp':0, 'lesion_fp':0, 'lesion_fn':0, 
        'gt_lesions':0, 'pred_lesions':0
    } for t in thresholds}
    
    vis_dir = base_dir / "artifacts" / "visualizations" / config['experiment_name'] / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, (imgs, masks, validities, names) in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
            img = imgs[0] # [3, H, W]
            mask = masks[0, 0].numpy() # [H, W]
            validity = validities[0, 0].item()
            name = names[0]
            
            if validity == 0:
                continue
                
            _, full_h, full_w = img.shape
            
            y_starts = list(range(0, full_h - native_patch_size + 1, stride))
            if y_starts[-1] + native_patch_size < full_h:
                y_starts.append(full_h - native_patch_size)
                
            x_starts = list(range(0, full_w - native_patch_size + 1, stride))
            if x_starts[-1] + native_patch_size < full_w:
                x_starts.append(full_w - native_patch_size)
                
            full_pred = torch.zeros(1, full_h, full_w, device=device)
            count_map = torch.zeros(1, full_h, full_w, device=device)
            
            for y in y_starts:
                for x in x_starts:
                    patch = img[:, y:y+native_patch_size, x:x+native_patch_size]
                    
                    patch_np = patch.permute(1, 2, 0).cpu().numpy()
                    patch_resized = cv2.resize(patch_np, (model_input_size, model_input_size), interpolation=cv2.INTER_LINEAR)
                    patch_t = torch.from_numpy(patch_resized).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                        logits = model(patch_t)
                    probs = torch.sigmoid(logits)[0] # [1, out_H, out_W]
                    
                    probs_np = probs.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                    probs_resized = cv2.resize(probs_np, (native_patch_size, native_patch_size), interpolation=cv2.INTER_LINEAR)
                    probs_t = torch.from_numpy(probs_resized).unsqueeze(0).to(device)
                    
                    full_pred[:, y:y+native_patch_size, x:x+native_patch_size] += probs_t
                    count_map[:, y:y+native_patch_size, x:x+native_patch_size] += 1
                    
            full_prob = (full_pred / count_map)[0].cpu().numpy() # [H, W]
            
            # Compute metrics for each threshold
            for t in thresholds:
                pred_mask = (full_prob > t).astype(np.uint8)
                
                # Pixel level
                p_b = pred_mask.astype(bool)
                g_b = mask.astype(bool)
                
                metrics_per_thresh[t]['pixel_tp'] += np.logical_and(p_b, g_b).sum()
                metrics_per_thresh[t]['pixel_fp'] += np.logical_and(p_b, np.logical_not(g_b)).sum()
                metrics_per_thresh[t]['pixel_fn'] += np.logical_and(np.logical_not(p_b), g_b).sum()
                metrics_per_thresh[t]['pixel_tn'] += np.logical_and(np.logical_not(p_b), np.logical_not(g_b)).sum()
                
                # Lesion level (distance tolerance = 20 pixels in native 4288x2848 coordinates)
                lesion_res = compute_lesion_metrics(pred_mask, mask, tolerance=20.0)
                metrics_per_thresh[t]['lesion_tp'] += lesion_res['tp']
                metrics_per_thresh[t]['lesion_fp'] += lesion_res['fp']
                metrics_per_thresh[t]['lesion_fn'] += lesion_res['fn']
                metrics_per_thresh[t]['gt_lesions'] += lesion_res['gt_count']
                metrics_per_thresh[t]['pred_lesions'] += lesion_res['pred_count']
                
            # Visualization at threshold 0.5
            if idx < 3:
                pred_05 = (full_prob > 0.5).astype(np.uint8)
                img_np = img.permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_np)
                axes[0].set_title(f'{name} Original')
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('GT MA Mask')
                axes[2].imshow(pred_05, cmap='gray')
                axes[2].set_title('Pred MA (th=0.5)')
                plt.tight_layout()
                plt.savefig(vis_dir / f"{name}_vis.png")
                plt.close(fig)

    print("\n=== MA Specialist FROC & Threshold Sweep ===")
    results = {}
    num_val_images = len(loader)
    
    best_dice = -1
    best_th = 0.5
    
    for t in thresholds:
        m = metrics_per_thresh[t]
        
        # Pixel
        ptp, pfp, pfn, ptn = m['pixel_tp'], m['pixel_fp'], m['pixel_fn'], m['pixel_tn']
        dice = (2 * ptp) / (2 * ptp + pfp + pfn + 1e-6)
        iou = ptp / (ptp + pfp + pfn + 1e-6)
        prec = ptp / (ptp + pfp + 1e-6)
        rec = ptp / (ptp + pfn + 1e-6)
        
        # Lesion
        ltp, lfp, lfn = m['lesion_tp'], m['lesion_fp'], m['lesion_fn']
        lprec = ltp / (ltp + lfp + 1e-6)
        lrec = ltp / (ltp + lfn + 1e-6)
        fpi = lfp / num_val_images
        
        results[t] = {
            'pixel_dice': float(dice),
            'pixel_prec': float(prec),
            'pixel_rec': float(rec),
            'lesion_prec': float(lprec),
            'lesion_rec': float(lrec),
            'lesion_fpi': float(fpi),
            'lesion_tp': int(ltp),
            'lesion_fp': int(lfp),
            'lesion_fn': int(lfn),
            'gt_lesions': int(m['gt_lesions'])
        }
        
        if dice > best_dice:
            best_dice = dice
            best_th = t
            
        print(f"Thresh {t:.2f} | P-Dice: {dice:.4f}, P-Prec: {prec:.4f}, P-Rec: {rec:.4f} | L-Sens: {lrec:.4f}, FPI: {fpi:.2f}")

    print(f"\nBest Validation Threshold (Pixel Dice): {best_th}")
    
    out_dir = base_dir / "artifacts/metrics" / config['experiment_name']
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}_froc_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default='val')
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split)
