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

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from final_project.data.idrid_segmentation import IDRiDSegmentationDataset
from final_project.models.unet import get_unet_model

out_dir = base_dir / "scratch" / "debug_unet"
out_dir.mkdir(parents=True, exist_ok=True)

def p(text):
    print(text)
    with open(out_dir / "diagnostic_report.txt", "a") as f:
        f.write(text + "\n")

if os.path.exists(out_dir / "diagnostic_report.txt"):
    os.remove(out_dir / "diagnostic_report.txt")

def main():
    config_path = base_dir / "final_project/configs/idrid_unet_effb0_ma_loss.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ds = IDRiDSegmentationDataset(
        config['data']['dataset_root'], 
        'val', 
        config['data']['split_json'],
        mode='val'
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    
    thresholds = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    class_keys = ['MA', 'HE', 'EX', 'SE']
    
    def evaluate_checkpoint(ckpt_path, do_qualitative=False):
        model = get_unet_model(encoder_name=config['model']['encoder_name'], num_classes=config['model']['num_classes']).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.eval()
        
        patch_size = 512
        overlap = 256
        stride = patch_size - overlap
        
        metrics_per_thresh = {t: {c: {'tp':0, 'fp':0, 'fn':0, 'tn':0, 'intersection':0, 'union':0, 'valid_imgs':0} for c in class_keys} for t in thresholds}
        
        vis_count = 0
        
        with torch.no_grad():
            for idx, (imgs, masks, validities, names) in enumerate(tqdm(loader, desc=f"Eval {Path(ckpt_path).name}")):
                img = imgs[0]
                mask = masks[0]
                validity = validities[0].numpy()
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
                        with torch.amp.autocast('cuda'):
                            logits = model(patch)
                        probs = torch.sigmoid(logits)[0]
                        full_pred[:, y:y+patch_size, x:x+patch_size] += probs
                        count_map[:, y:y+patch_size, x:x+patch_size] += 1
                        
                full_prob = (full_pred / count_map).cpu().numpy()
                gt_mask = mask.numpy()
                
                # Threshold Sweeps
                for t in thresholds:
                    pred_mask = (full_prob > t)
                    for c in range(4):
                        if validity[c] > 0:
                            metrics_per_thresh[t][class_keys[c]]['valid_imgs'] += 1
                            p_mask = pred_mask[c].astype(bool)
                            g_mask = gt_mask[c].astype(bool)
                            
                            tp = np.logical_and(p_mask, g_mask).sum()
                            fp = np.logical_and(p_mask, np.logical_not(g_mask)).sum()
                            fn = np.logical_and(np.logical_not(p_mask), g_mask).sum()
                            tn = np.logical_and(np.logical_not(p_mask), np.logical_not(g_mask)).sum()
                            
                            metrics_per_thresh[t][class_keys[c]]['tp'] += tp
                            metrics_per_thresh[t][class_keys[c]]['fp'] += fp
                            metrics_per_thresh[t][class_keys[c]]['fn'] += fn
                            metrics_per_thresh[t][class_keys[c]]['tn'] += tn
                            metrics_per_thresh[t][class_keys[c]]['intersection'] += tp
                            metrics_per_thresh[t][class_keys[c]]['union'] += (p_mask.sum() + g_mask.sum() - tp)

                # Qualitative Visualizations
                if do_qualitative and vis_count < 3 and validity[0] > 0:
                    img_np = img.permute(1, 2, 0).numpy()
                    img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                    
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    axes[0].imshow(img_np)
                    axes[0].set_title('Original Fundus')
                    axes[1].imshow(gt_mask[0], cmap='gray')
                    axes[1].set_title('MA GT')
                    axes[2].imshow(full_prob[0], cmap='viridis')
                    axes[2].set_title('Raw MA Prob')
                    axes[3].imshow(full_prob[0] > 0.5, cmap='gray')
                    axes[3].set_title('Threshold 0.5')
                    plt.savefig(out_dir / f"{name}_MA_diagnostic.png")
                    plt.close(fig)
                    
                    if validity[2] > 0:
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        axes[0].imshow(img_np)
                        axes[0].set_title('Original Fundus')
                        axes[1].imshow(gt_mask[2], cmap='gray')
                        axes[1].set_title('EX GT')
                        axes[2].imshow(full_prob[2], cmap='viridis')
                        axes[2].set_title('Raw EX Prob')
                        axes[3].imshow(full_prob[2] > 0.5, cmap='gray')
                        axes[3].set_title('Threshold 0.5')
                        plt.savefig(out_dir / f"{name}_EX_diagnostic.png")
                        plt.close(fig)

                    if validity[1] > 0:
                        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                        axes[0].imshow(img_np)
                        axes[0].set_title('Original Fundus')
                        axes[1].imshow(gt_mask[1], cmap='gray')
                        axes[1].set_title('HE GT')
                        axes[2].imshow(full_prob[1], cmap='viridis')
                        axes[2].set_title('Raw HE Prob')
                        axes[3].imshow(full_prob[1] > 0.5, cmap='gray')
                        axes[3].set_title('Threshold 0.5')
                        plt.savefig(out_dir / f"{name}_HE_diagnostic.png")
                        plt.close(fig)
                        
                    vis_count += 1
                    
        return metrics_per_thresh
        
    best_ckpt = base_dir / "artifacts/checkpoints/idrid_unet_effb0_ma_loss/best.pt"
    
    p("=== UNET PER-CLASS THRESHOLD SWEEP (best.pt) ===")
    metrics_thresh = evaluate_checkpoint(best_ckpt, do_qualitative=True)
    
    best_dice_per_class = {c: -1 for c in class_keys}
    best_thresh_per_class = {c: -1 for c in class_keys}
    
    for c in class_keys:
        p(f"\n--- {c} Thresholds ---")
        for t in thresholds:
            m = metrics_thresh[t][c]
            if m['valid_imgs'] == 0: continue
            
            dice = (2 * m['intersection']) / (2 * m['intersection'] + m['fp'] + m['fn'] + 1e-6)
            iou = m['intersection'] / (m['union'] + 1e-6)
            prec = m['tp'] / (m['tp'] + m['fp'] + 1e-6)
            rec = m['tp'] / (m['tp'] + m['fn'] + 1e-6)
            
            if dice > best_dice_per_class[c]:
                best_dice_per_class[c] = dice
                best_thresh_per_class[c] = t
                
            p(f"Thresh {t:.2f} | Dice: {dice:.4f}, IoU: {iou:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")
            
        p(f"Best Threshold for {c}: {best_thresh_per_class[c]} (Dice: {best_dice_per_class[c]:.4f})")

if __name__ == "__main__":
    main()
