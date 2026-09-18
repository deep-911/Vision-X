import argparse
import yaml
import json
import os
import time
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Fix path to import modules
import sys
base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.data.registry import get_dataset
from mvp.models.baselines import BaselineClassification, BaselineOrdinal
from mvp.training.loss import CORALLoss, coral_decoding
from mvp.evaluation.metrics import calculate_classification_metrics

def get_git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(base_dir)).decode('utf-8').strip()
    except Exception:
        return "unknown"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic algorithms if needed
    
def get_loaders(config, base_dir):
    if 'data' in config:
        # New generic data loading via registry
        dataset_name = config['data']['dataset_name']
        train_split = config['data'].get('train_split', 'train')
        val_split = config['data'].get('val_split', 'val')
        batch_size = config['dataset']['batch_size']
        num_workers = config['dataset'].get('num_workers', 0)
        
        train_ds = get_dataset(dataset_name, train_split, config, augment=True)
        val_ds = get_dataset(dataset_name, val_split, config, augment=False)
    else:
        # Legacy IDRiD hardcoded paths for MVP reproducibility
        split_json = base_dir / "mvp/data/splits/idrid_split_v1.json"
        img_dir_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
        img_dir_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
        labels_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
        labels_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
        
        img_size = tuple(config['dataset']['img_size'])
        batch_size = config['dataset']['batch_size']
        num_workers = config['dataset'].get('num_workers', 0)
        
        # We need a mock config for the registry for legacy IDRiD
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
        train_ds = get_dataset('idrid', 'train', mock_config, augment=True)
        val_ds = get_dataset('idrid', 'val', mock_config, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def main(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    set_seed(config['training']['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    run_id = config['experiment_name']
    metrics_dir = base_dir / config['training']['metrics_dir'] / run_id
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_dir = base_dir / config['training']['checkpoint_dir'] / run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader = get_loaders(config, base_dir)
    
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    
    if model_type == 'classification':
        model = BaselineClassification(num_classes=num_classes).to(device)
        class_weights = None
        if config.get('data', {}).get('imbalance_strategy') == 'class_weights':
            try:
                import numpy as np
                if hasattr(train_loader.dataset, 'df') and hasattr(train_loader.dataset, 'label_col'):
                    labels = train_loader.dataset.df[train_loader.dataset.label_col].values
                elif hasattr(train_loader.dataset, 'df_train'):
                    labels = train_loader.dataset.df_train['Retinopathy grade'].values
                else:
                    labels = []
                
                if len(labels) > 0:
                    class_counts = np.bincount(labels)
                    weights = len(labels) / (len(class_counts) * class_counts)
                    class_weights = torch.FloatTensor(weights).to(device)
                    print(f"Applying class weights: {class_weights}")
            except Exception as e:
                print(f"Failed to calculate class weights: {e}")
                
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif model_type == 'ordinal':
        model = BaselineOrdinal(num_classes=num_classes).to(device)
        criterion = CORALLoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    best_val_qwk = -1.0
    
    start_time = time.time()
    
    for epoch in range(config['training']['epochs']):
        model.train()
        train_loss = 0.0
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                logits = model(imgs)
                loss = criterion(logits, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * imgs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                    
                val_loss += loss.item() * imgs.size(0)
                
                if model_type == 'classification':
                    probas = torch.softmax(logits, dim=1)
                else: # ordinal
                    # approximate class probabilities for metrics calculation
                    # The pure decoding is coral_decoding(logits)
                    preds = coral_decoding(logits)
                    # We create dummy one-hot probs for metrics to work identically, 
                    # or better: metric function needs raw predictions if ordinal
                    probas = torch.zeros(logits.size(0), num_classes, device=device)
                    probas.scatter_(1, preds.unsqueeze(1), 1.0)
                    
                all_preds.append(probas.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        
        import numpy as np
        all_preds = np.vstack(all_preds)
        all_labels = np.concatenate(all_labels)
        
        metrics = calculate_classification_metrics(all_labels, all_preds)
        qwk = metrics['qwk']
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val QWK: {qwk:.4f}")
        
        # Save latest
        torch.save(model.state_dict(), ckpt_dir / "latest.pt")
        
        # Save best
        if qwk > best_val_qwk:
            best_val_qwk = qwk
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print("  --> Saved new best checkpoint!")
            
    total_time = time.time() - start_time
    
    # Save artifacts
    run_info = {
        'run_id': run_id,
        'git_commit': get_git_commit(),
        'best_val_qwk': best_val_qwk,
        'total_time_seconds': total_time,
        'config': config
    }
    with open(metrics_dir / "run_info.json", 'w') as f:
        json.dump(run_info, f, indent=4)
        
    print(f"Training complete. Best Val QWK: {best_val_qwk:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
