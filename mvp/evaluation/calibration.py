import os
import sys
import yaml
import json
import torch
from torch import nn, optim
import numpy as np
import argparse
from pathlib import Path
from sklearn.metrics import brier_score_loss

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.data.registry import get_dataset
from mvp.models.baselines import BaselineClassification
from torch.utils.data import DataLoader

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

def calc_ece(probs, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

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

    all_logits = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                logits = model(imgs)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Raw ECE and Brier
    raw_probs = torch.softmax(all_logits, dim=1).cpu().numpy()
    labels_np = all_labels.cpu().numpy()
    
    raw_ece = calc_ece(raw_probs, labels_np)
    
    # Brier score (multi-class one-hot)
    labels_one_hot = np.eye(num_classes)[labels_np]
    raw_brier = np.mean(np.sum((raw_probs - labels_one_hot)**2, axis=1))

    print(f"Raw ECE: {raw_ece:.4f}")
    print(f"Raw Brier Score: {raw_brier:.4f}")

    # Fit temperature scaling
    temperature = nn.Parameter(torch.ones(1, device=device))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = criterion(all_logits / temperature, all_labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    final_temp = temperature.item()
    print(f"Optimal Temperature: {final_temp:.4f}")

    # Calibrated ECE and Brier
    cal_probs = torch.softmax(all_logits / final_temp, dim=1).cpu().numpy()
    cal_ece = calc_ece(cal_probs, labels_np)
    cal_brier = np.mean(np.sum((cal_probs - labels_one_hot)**2, axis=1))

    print(f"Calibrated ECE: {cal_ece:.4f}")
    print(f"Calibrated Brier Score: {cal_brier:.4f}")

    out_dir = base_dir / "artifacts/reports/aptos_calibration" / config['experiment_name']
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "temperature": final_temp,
        "raw_ece": float(raw_ece),
        "raw_brier": float(raw_brier),
        "calibrated_ece": float(cal_ece),
        "calibrated_brier": float(cal_brier)
    }
    
    with open(out_dir / "calibration.json", 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved calibration to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
