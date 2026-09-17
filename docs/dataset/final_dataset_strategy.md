# Final Dataset Strategy for VisionX

This document outlines the final data strategy for the SIH26038 VisionX architecture, ensuring the correct datasets are strictly mapped to their intended modular tasks.

## 1. APTOS 2019 (Blindness Detection)
- **Purpose**: Primary dataset for the core DR Severity Grading module (Grades 0-4).
- **Label Types**: 5-class ordinal grading (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative).
- **Image Counts**: 3,662 officially labeled training images. 
- **Pipeline Integration**: Serves as the central training and validation corpus for the EfficientNet/Softmax grading engine.
- **Evaluation Strategy**:
  - **Training / Validation**: Internal splits (e.g., 80/10/10) will be used to train and select the final model hyper-parameters and thresholds.
  - **Note on Official Test Set**: The public APTOS test set is a Kaggle competition holdout lacking public ground-truth labels. Therefore, test-set inference cannot be evaluated locally; we rely on cross-dataset evaluation.
- **Licensing/Notes**: Available for research and non-commercial purposes. 

## 2. IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Purpose**: Secondary cross-dataset grading evaluation and primary lesion-level supervision.
- **Label Types**: 5-class DR Grading, Lesion Segmentation masks (MA, HE, SE, EX), and Optic Disc / Fovea coordinates.
- **Pipeline Integration**: 
  - **Cross-Dataset Evaluation**: The locked 103-image IDRiD test set will be used to evaluate the generalized performance of the APTOS-trained grading model.
  - **Lesion Detection**: Serves as the primary training ground for the lesion segmentation branch (SegFormer).
- **Licensing/Notes**: CC BY 4.0. High-quality but extremely small sample size for classification training (413 images).

## 3. DRIVE (Digital Retinal Images for Vessel Extraction)
- **Purpose**: Retinal vessel segmentation and extraction.
- **Label Types**: Binary pixel-level vessel masks.
- **Pipeline Integration**: Used exclusively to train the structural evidence branch (U-Net/SegFormer variant) that feeds structural features into the Trust Engine.
- **Licensing/Notes**: Academic research only. 

## 4. Messidor-2
- **Purpose**: External robustness and domain-shift evaluation for the final architecture.
- **Label Types**: 5-class DR Grading (via secondary independent annotation protocols, as the base dataset does not ship with official ground-truth labels).
- **Pipeline Integration**: Used purely for final architectural stress-testing and Out-Of-Distribution (OOD) verification.
- **Licensing/Notes**: Requires registered access. Care must be taken to ensure the utilized labels come from a scientifically valid peer-reviewed consensus (e.g., the standard Kaggle/Decathlon annotations).

## 5. EyePACS (Optional / Future)
- **Purpose**: Not on the critical path. Will only be used for optional future scaling if the hardware environment (Google Colab) permits training on 35,000+ images.
