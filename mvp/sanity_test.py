import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))

from mvp.data.datasets.idrid_grading import IDRiDGradingDataset
from mvp.models.baselines import BaselineClassification

def main():
    print("--- Starting Sanity Test Run ---")
    
    with open(base_dir / "mvp/configs/mvp_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    split_json = base_dir / "mvp/data/splits/idrid_split_v1.json"
    img_dir_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    img_dir_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    labels_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    labels_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    
    # Only load a tiny subset (e.g., just the first 8 items) for the sanity test.
    # To do this safely without modifying the dataset class, we can just break the loop early.
    train_ds = IDRiDGradingDataset("train", split_json, img_dir_train, img_dir_test, labels_train, labels_test, img_size=(224, 224), augment=True)
    train_dl = DataLoader(train_ds, batch_size=config['dataset']['batch_size'], shuffle=True)
    
    model = BaselineClassification(num_classes=config['model']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scaler = torch.amp.GradScaler('cuda', enabled=config['training']['mixed_precision'])
    
    # Just 1 epoch, 2 batches max for sanity
    model.train()
    for epoch in range(1):
        for i, (imgs, labels, names) in enumerate(train_dl):
            if i >= 2: break
            
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=config['training']['mixed_precision']):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")
            
    print("Sanity test forward/backward pass completed successfully!")
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"Max GPU Memory Allocated: {max_mem:.2f} MB")
        
    # Save a fake checkpoint
    ckpt_path = base_dir / "artifacts/checkpoints/sanity_ckpt.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    main()
