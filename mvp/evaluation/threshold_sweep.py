import os
import sys
import yaml
import json
import torch
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, auc

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.data.registry import get_dataset
from mvp.models.baselines import BaselineClassification
from torch.utils.data import DataLoader

def main(config_path, checkpoint_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validation only
    dataset_name = config['data']['dataset_name']
    val_split = config['data'].get('val_split', 'val')
    img_size = tuple(config['data'].get('img_size', [224, 224]))
    batch_size = config['dataset']['batch_size']
    num_workers = config['dataset'].get('num_workers', 0)

    val_ds = get_dataset(dataset_name, val_split, config, augment=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = config['model']['num_classes']
    model = BaselineClassification(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                logits = model(imgs)
            probas = torch.softmax(logits, dim=1)
            all_preds.append(probas.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)

    # Referable DR is grade >= 2
    referable_labels = (all_labels >= 2).astype(int)
    referable_probs = np.sum(all_preds[:, 2:], axis=1)

    # Calculate ROC and PR curves
    fpr, tpr, thresholds = roc_curve(referable_labels, referable_probs)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(referable_labels, referable_probs)
    pr_auc = auc(recall, precision)
    
    print(f"Referable ROC-AUC: {roc_auc:.4f}")
    print(f"Referable PR-AUC: {pr_auc:.4f}")

    # Sweep thresholds to find sensitivity >= 0.90 and specificity >= 0.85
    # specificity = 1 - fpr, sensitivity = tpr
    specificities = 1 - fpr
    sensitivities = tpr

    feasible_idx = np.where((sensitivities >= 0.90) & (specificities >= 0.85))[0]
    
    threshold_table = []
    for idx in range(len(thresholds)):
        if idx % max(1, len(thresholds)//20) == 0 or idx in feasible_idx:
            threshold_table.append({
                "threshold": float(thresholds[idx]),
                "sensitivity": float(sensitivities[idx]),
                "specificity": float(specificities[idx]),
                "feasible": bool(idx in feasible_idx)
            })

    if len(feasible_idx) > 0:
        # Criterion: maximize specificity subject to sens>=0.90 and spec>=0.85
        best_local_idx = np.argmax(specificities[feasible_idx])
        best_idx = feasible_idx[best_local_idx]
        best_threshold = float(thresholds[best_idx])
        print(f"Found {len(feasible_idx)} feasible thresholds.")
        print(f"Selected Threshold: {best_threshold:.4f} (Sens: {sensitivities[best_idx]:.4f}, Spec: {specificities[best_idx]:.4f})")
    else:
        print("No threshold satisfies Sens >= 0.90 and Spec >= 0.85.")
        # Fallback to maximizing Sens + Spec (Youden's J statistic)
        j_stat = sensitivities + specificities - 1
        best_idx = np.argmax(j_stat)
        best_threshold = float(thresholds[best_idx])
        print(f"Fallback Threshold: {best_threshold:.4f} (Sens: {sensitivities[best_idx]:.4f}, Spec: {specificities[best_idx]:.4f})")

    out_dir = base_dir / "artifacts/reports/aptos_threshold_analysis" / config['experiment_name']
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "best_threshold": best_threshold,
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "sensitivity": float(sensitivities[best_idx]),
        "specificity": float(specificities[best_idx]),
        "feasible": bool(len(feasible_idx) > 0),
        "threshold_table": threshold_table
    }
    
    with open(out_dir / "threshold_analysis.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved threshold analysis to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
