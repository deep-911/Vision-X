# MVP Reproducibility Package

This document serves as the machine-readable experiment record for the MVP baselines.

## System & Environment
- **Python Version**: 3.14 (implied by `cp314` wheels in previous logs)
- **PyTorch Version**: 2.14.0+cu130
- **Torchvision Version**: 0.29.0+cu130
- **CUDA Version**: 13.0
- **GPU**: NVIDIA RTX 3050 Laptop GPU (4 GB)
- **Git Commit**: `afe97c7` (Development Branch)

## Data Protocol
- **Dataset Split Identifier**: IDRiD Disease Grading (Official Split)
- **Train/Val/Test Counts**: 329 Train / 84 Validation / 103 Test
- **Preprocessing**: Resize to 224x224, ImageNet normalization.

## Model Configurations

### Baseline A (Softmax)
- **Model Definition**: `EfficientNet-B0` with 5-class `Linear` head
- **Pretrained Weights**: `EfficientNet_B0_Weights.DEFAULT`
- **Optimizer**: AdamW (lr=0.0005, wd=0.0001)
- **Scheduler**: None
- **Batch Size**: 16
- **Image Size**: 224x224
- **AMP**: Enabled (float16)
- **Epochs**: 15
- **Random Seeds**: 42 (PyTorch, NumPy, Random)

### Baseline B (CORAL Ordinal)
- **Model Definition**: `EfficientNet-B0` with `CoralLayer` (Shared Weight, 4 Independent Rank Biases)
- **Pretrained Weights**: `EfficientNet_B0_Weights.DEFAULT`
- **Optimizer**: AdamW (lr=0.0005, wd=0.0001)
- **Scheduler**: None
- **Batch Size**: 16
- **Image Size**: 224x224
- **AMP**: Enabled (float16)
- **Epochs**: 15
- **Random Seeds**: 42 (PyTorch, NumPy, Random)

## Evaluation Protocol
- **Validation Protocol**: Evaluated at the end of every epoch. 
- **Checkpoint Selection Rule**: Best model selected by highest **Validation QWK**.
- **Test Protocol**: Held-out 103 images evaluated exactly once using the frozen best-validation checkpoint. No hyperparameters or thresholds were adjusted based on test set results.
