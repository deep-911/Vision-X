# APTOS Provenance Audit

## Provenance Conclusion
The dataset currently residing at `Datasets/aptos/raw_aptos/` is a **project-derived or third-party pre-split version** of the official Kaggle APTOS 2019 labeled training corpus.

**It is NOT the official Kaggle competition train/test split.**

### Evidence
1. **Total Labeled Images**: The official Kaggle APTOS 2019 dataset released exactly 3,662 labeled images in its public `train` corpus. The dataset present locally contains exactly 3,662 labeled images in total across all its subsets.
2. **Missing Official Test Set**: The official Kaggle `test` corpus contained 1,928 images with no public ground truth. This corpus is entirely absent from the local files.
3. **Split Ratio**: The 2,930 / 366 / 366 partitioning mathematically represents an exact 80% / 10% / 10% split of the 3,662 official training images.
4. **File Naming**: The files `train_1.csv` and `valid.csv` are non-standard. The Kaggle original was a single `train.csv`.

## Split Protocol
To maintain scientific clarity and prevent confusion with the Kaggle competition leaderboard protocol:
- This 2930/366/366 structure will hereafter be referred to strictly as the **"APTOS Project Split"**.
- **2930 Split (`train_1.csv`)** $\rightarrow$ **Training**: Used for gradient updates and model fitting.
- **366 Split (`valid.csv`)** $\rightarrow$ **Validation**: Used for epoch-wise checkpoint selection, hyperparameter tuning, and early stopping.
- **366 Split (`test.csv`)** $\rightarrow$ **Internal Holdout**: This is a strictly locked evaluation set. It must remain entirely untouched until the final APTOS model-development protocol is completely frozen. It will serve as the final intra-dataset performance benchmark.

## Dataset Integrity Report
- **Total Images**: 3,662
- **Image Format**: `.png`
- **Missing / Orphans**: 0 missing images, 0 orphan labels. Perfect 1-to-1 mapping.
- **Grade Range**: 0-4 (No DR to Proliferative DR). No invalid grades detected.
- **Duplicate Detection**: No filename collisions exist. (Pixel-wise deduplication will be delegated to future dataloader augmentation phases if computationally necessary, but filenames are unique).
- **Total Disk Size**: ~8.6 GB
- **Per-Split Image Counts & Distribution**:
  - **Train (2,930)**: `0: 1434, 1: 300, 2: 808, 3: 154, 4: 234`
  - **Validation (366)**: `0: 172, 1: 40, 2: 104, 3: 22, 4: 28`
  - **Holdout (366)**: `0: 199, 1: 30, 2: 87, 3: 17, 4: 33`
