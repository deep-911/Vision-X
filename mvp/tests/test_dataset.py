import sys
from pathlib import Path
import yaml
import torch
from torch.utils.data import DataLoader

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.data.datasets.idrid_grading import IDRiDGradingDataset

def test():
    print("Running dataset sanity checks...")
    split_json = base_dir / "mvp/data/splits/idrid_split_v1.json"
    img_dir_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/a. Training Set"
    img_dir_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/1. Original Images/b. Testing Set"
    labels_train = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    labels_test = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    
    ds = IDRiDGradingDataset("train", split_json, img_dir_train, img_dir_test, labels_train, labels_test, img_size=(224, 224), augment=True)
    
    dl = DataLoader(ds, batch_size=4, shuffle=True)
    imgs, labels, names = next(iter(dl))
    
    assert imgs.shape == (4, 3, 224, 224), f"Shape mismatch: {imgs.shape}"
    assert imgs.dtype == torch.float32, f"Dtype mismatch: {imgs.dtype}"
    assert not torch.isnan(imgs).any(), "NaNs in images"
    assert labels.min() >= 0 and labels.max() <= 4, "Labels out of range"
    
    print("Sanity checks passed! Batch shape:", imgs.shape, "Labels:", labels)

if __name__ == "__main__":
    test()
