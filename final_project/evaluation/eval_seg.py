import os
import sys
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset
from final_project.models.segformer import SegformerMultiLabel

def evaluate(config_path, checkpoint_path, split='val'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        split, 
        config['data']['split_json'],
        mode='val' # sliding window mode loads full image
    )
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=config['dataset']['num_workers'])

    # Model
    model = SegformerMultiLabel(num_classes=config['model']['num_classes']).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    patch_size = 512
    overlap = 256
    stride = patch_size - overlap
    
    metrics = {
        'MA': {'tp':0, 'fp':0, 'fn':0, 'tn':0, 'intersection':0, 'union':0, 'valid_imgs':0},
        'HE': {'tp':0, 'fp':0, 'fn':0, 'tn':0, 'intersection':0, 'union':0, 'valid_imgs':0},
        'EX': {'tp':0, 'fp':0, 'fn':0, 'tn':0, 'intersection':0, 'union':0, 'valid_imgs':0},
        'SE': {'tp':0, 'fp':0, 'fn':0, 'tn':0, 'intersection':0, 'union':0, 'valid_imgs':0}
    }
    
    class_keys = ['MA', 'HE', 'EX', 'SE']
    
    vis_dir = base_dir / "artifacts" / "visualizations" / config['experiment_name'] / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx, (imgs, masks, validities, names) in enumerate(tqdm(loader, desc=f"Evaluating {split}")):
            img = imgs[0] # [3, H, W]
            mask = masks[0] # [4, H, W]
            validity = validities[0] # [4]
            name = names[0]
            
            _, full_h, full_w = img.shape
            
            y_starts = list(range(0, full_h - patch_size + 1, stride))
            if y_starts[-1] + patch_size < full_h:
                y_starts.append(full_h - patch_size)
                
            x_starts = list(range(0, full_w - patch_size + 1, stride))
            if x_starts[-1] + patch_size < full_w:
                x_starts.append(full_w - patch_size)
                
            full_pred = torch.zeros(4, full_h, full_w, device=device)
            count_map = torch.zeros(4, full_h, full_w, device=device)
            
            for y in y_starts:
                for x in x_starts:
                    patch = img[:, y:y+patch_size, x:x+patch_size].unsqueeze(0).to(device)
                    with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                        logits = model(patch)
                    probs = torch.sigmoid(logits)[0] # [4, 512, 512]
                    
                    full_pred[:, y:y+patch_size, x:x+patch_size] += probs
                    count_map[:, y:y+patch_size, x:x+patch_size] += 1
                    
            full_pred = full_pred / count_map # average overlaps
            
            # Binary predictions
            pred_mask = (full_pred > 0.5).cpu().numpy()
            gt_mask = mask.numpy()
            val_arr = validity.numpy()
            
            # Save qualitative visualization (max 3 images)
            if idx < 3:
                img_np = img.permute(1, 2, 0).numpy()
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = np.clip(img_np, 0, 1)
                
                fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                for c in range(4):
                    axes[c, 0].imshow(img_np)
                    axes[c, 0].set_title(f'{name} Original')
                    axes[c, 1].imshow(gt_mask[c], cmap='gray')
                    axes[c, 1].set_title(f'GT {class_keys[c]} (Valid={val_arr[c]})')
                    axes[c, 2].imshow(pred_mask[c], cmap='gray')
                    axes[c, 2].set_title(f'Pred {class_keys[c]}')
                plt.tight_layout()
                plt.savefig(vis_dir / f"{name}_vis.png")
                plt.close(fig)
            
            # Compute metrics only for valid classes
            for c, cls_name in enumerate(class_keys):
                if val_arr[c] > 0:
                    metrics[cls_name]['valid_imgs'] += 1
                    
                    p = pred_mask[c].astype(bool)
                    g = gt_mask[c].astype(bool)
                    
                    tp = np.logical_and(p, g).sum()
                    fp = np.logical_and(p, np.logical_not(g)).sum()
                    fn = np.logical_and(np.logical_not(p), g).sum()
                    tn = np.logical_and(np.logical_not(p), np.logical_not(g)).sum()
                    
                    metrics[cls_name]['tp'] += tp
                    metrics[cls_name]['fp'] += fp
                    metrics[cls_name]['fn'] += fn
                    metrics[cls_name]['tn'] += tn
                    
                    metrics[cls_name]['intersection'] += tp
                    metrics[cls_name]['union'] += (p.sum() + g.sum() - tp)

    # Aggregate metrics
    results = {}
    macro_dice = 0
    macro_iou = 0
    macro_sens = 0
    macro_spec = 0
    macro_prec = 0
    
    for cls_name in class_keys:
        tp = metrics[cls_name]['tp']
        fp = metrics[cls_name]['fp']
        fn = metrics[cls_name]['fn']
        tn = metrics[cls_name]['tn']
        intersection = metrics[cls_name]['intersection']
        union = metrics[cls_name]['union']
        valid = metrics[cls_name]['valid_imgs']
        
        dice = (2 * intersection) / (2 * intersection + fp + fn + 1e-6)
        iou = intersection / (union + 1e-6)
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        prec = tp / (tp + fp + 1e-6)
        
        results[cls_name] = {
            'valid_images': valid,
            'dice': float(dice),
            'iou': float(iou),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'precision': float(prec)
        }
        
        macro_dice += dice
        macro_iou += iou
        macro_sens += sens
        macro_spec += spec
        macro_prec += prec
        
    results['Macro'] = {
        'dice': float(macro_dice / 4),
        'iou': float(macro_iou / 4),
        'sensitivity': float(macro_sens / 4),
        'specificity': float(macro_spec / 4),
        'precision': float(macro_prec / 4)
    }
    
    print("\n=== Final Segmentation Metrics ===")
    print(json.dumps(results, indent=4))
    
    out_dir = base_dir / "artifacts/metrics" / config['experiment_name']
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{split}_seg_metrics.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default='val')
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.split)
