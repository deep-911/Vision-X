import os
import glob
import json
import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

def main():
    train_imgs = glob.glob('Datasets/IDRiD_extracted/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set/*.jpg')
    masks_dir = 'Datasets/IDRiD_extracted/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set'
    
    data = []
    labels = []
    for p in sorted(train_imgs):
        img_name = os.path.basename(p)
        base = os.path.splitext(img_name)[0]
        
        ma = int(os.path.exists(f'{masks_dir}/1. Microaneurysms/{base}_MA.tif'))
        he = int(os.path.exists(f'{masks_dir}/2. Haemorrhages/{base}_HE.tif'))
        ex = int(os.path.exists(f'{masks_dir}/3. Hard Exudates/{base}_EX.tif'))
        se = int(os.path.exists(f'{masks_dir}/4. Soft Exudates/{base}_SE.tif'))
        
        data.append([base])
        labels.append([ma, he, ex, se])
        
    data = np.array(data)
    labels = np.array(labels)
    
    # 54 total, we want 10 val, so test_size = 10/54 = 0.185
    # iterative_train_test_split expects 2D array for features, which we have
    X_train, y_train, X_val, y_val = iterative_train_test_split(data, labels, test_size=0.185)
    
    # Verify counts
    print(f"Train samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    
    # Print presence
    print("Train presence (MA, HE, EX, SE):", y_train.sum(axis=0))
    print("Val presence (MA, HE, EX, SE):", y_val.sum(axis=0))
    
    split_dict = {
        "train": [x[0] for x in X_train],
        "val": [x[0] for x in X_val]
    }
    
    os.makedirs('final_project/data/splits', exist_ok=True)
    with open('final_project/data/splits/idrid_seg_split.json', 'w') as f:
        json.dump(split_dict, f, indent=4)
        
if __name__ == "__main__":
    main()
