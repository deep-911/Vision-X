import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

import sys
base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.data.registry import get_dataset
from mvp.models.baselines import BaselineClassification, BaselineOrdinal
from mvp.training.loss import coral_decoding
from mvp.evaluation.metrics import calculate_classification_metrics

def main(config_path, split, ckpt_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if split == 'test':
        print("\n" + "="*50)
        print("WARNING: Evaluating on the LOCKED official test set.")
        print("Ensure this is only done exactly once per finalized baseline.")
        print("="*50 + "\n")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if 'data' in config:
        dataset_name = config['data']['dataset_name']
        
        # Override split based on config mappings if needed, else use passed split directly
        if split == 'train':
            actual_split = config['data'].get('train_split', 'train')
        elif split == 'val':
            actual_split = config['data'].get('val_split', 'val')
        elif split == 'test':
            actual_split = config['data'].get('holdout_split', 'test')
        else:
            actual_split = split
            
        img_size = tuple(config['data'].get('img_size', [224, 224]))
        batch_size = config['dataset']['batch_size']
        num_workers = config['dataset'].get('num_workers', 0)
        
        ds = get_dataset(dataset_name, actual_split, config, augment=False)
    else:
        split_json = base_dir / "mvp/data/splits/idrid_split_v1.json"
        img_dir_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
        img_dir_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
        labels_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
        labels_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
        
        img_size = tuple(config['dataset']['img_size'])
        batch_size = config['dataset']['batch_size']
        num_workers = config['dataset'].get('num_workers', 0)
        
        mock_config = {
            'data': {
                'split_json_path': str(split_json),
                'img_dir_train': str(img_dir_train),
                'img_dir_test': str(img_dir_test),
                'labels_train_csv': str(labels_train),
                'labels_test_csv': str(labels_test),
                'img_size': img_size
            }
        }
        ds = get_dataset('idrid', split, mock_config, augment=False)
        
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    
    if model_type == 'classification':
        model = BaselineClassification(num_classes=num_classes).to(device)
    elif model_type == 'ordinal':
        model = BaselineOrdinal(num_classes=num_classes).to(device)
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                logits = model(imgs)
                
            if model_type == 'classification':
                probas = torch.softmax(logits, dim=1)
            else:
                preds = coral_decoding(logits)
                probas = torch.zeros(logits.size(0), num_classes, device=device)
                probas.scatter_(1, preds.unsqueeze(1), 1.0)
                
            all_preds.append(probas.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    import numpy as np
    all_preds = np.vstack(all_preds)
    all_labels = np.concatenate(all_labels)
    
    metrics = calculate_classification_metrics(all_labels, all_preds)
    
    out_dir = base_dir / config['training']['metrics_dir'] / config['experiment_name']
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{split}_metrics.json"
    
    with open(out_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\n{split.capitalize()} Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
            
    print(f"\nSaved metrics to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, choices=['val', 'test'], required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.split, args.checkpoint)
