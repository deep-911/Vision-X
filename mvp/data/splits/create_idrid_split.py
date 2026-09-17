import pandas as pd
import json
import random
from pathlib import Path

def create_splits():
    base_dir = Path(__file__).resolve().parents[3]
    train_csv = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
    test_csv = base_dir / "Datasets/IDRiD_extracted/B. Disease Grading/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"
    
    df_train = pd.read_csv(train_csv).dropna(how='all', axis=1)
    df_test = pd.read_csv(test_csv).dropna(how='all', axis=1)
    
    # Stratified split of training set into train/val (80/20)
    # Using fixed seed for deterministic behavior
    random.seed(42)
    
    train_split = []
    val_split = []
    
    for grade in sorted(df_train['Retinopathy grade'].unique()):
        grade_df = df_train[df_train['Retinopathy grade'] == grade]
        images = grade_df['Image name'].tolist()
        random.shuffle(images)
        
        split_idx = int(len(images) * 0.8)
        train_split.extend(images[:split_idx])
        val_split.extend(images[split_idx:])
        
    test_split = df_test['Image name'].tolist()
    
    manifest = {
        "train": train_split,
        "val": val_split,
        "test": test_split
    }
    
    split_file = Path(__file__).resolve().parent / "idrid_split_v1.json"
    with open(split_file, 'w') as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Split created: Train={len(train_split)}, Val={len(val_split)}, Test={len(test_split)}")
    print(f"Saved to {split_file}")

if __name__ == "__main__":
    create_splits()
