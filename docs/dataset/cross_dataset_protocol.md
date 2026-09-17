# Cross-Dataset Protocol

To ensure robust evaluation and prevent data leakage, the SIH26038 architecture utilizes multiple independent datasets, strictly compartmentalized by their intended physiological or evaluation target.

## 1. APTOS (APTOS 2019 Blindness Detection)
- **Role**: **Main Grading Model Development**
- **Function**: The APTOS Project Split (2930/366/366) serves as the primary engine for training the global 5-class Diabetic Retinopathy severity classification model (EfficientNet backbone).

## 2. IDRiD (Indian Diabetic Retinopathy Image Dataset)
- **Role 1**: **Lesion Supervision**. The localized pixel-level masks will be used to train the lesion segmentation branch (SegFormer) for Microaneurysms, Hemorrhages, Hard Exudates, and Soft Exudates.
- **Role 2**: **Cross-Dataset Grading Evaluation**. The frozen IDRiD grading test set (103 images) will be used to benchmark the APTOS-trained model on a completely unseen geographical and hardware domain.
- **Role 3**: **Lesion Evaluation**. The lesion branch will be evaluated against the IDRiD test masks.

## 3. DRIVE (Digital Retinal Images for Vessel Extraction)
- **Role**: **Vessel Segmentation**
- **Function**: Used exclusively to train and evaluate the retinal vessel extraction pipeline, which feeds structural graph evidence into the final Trust Engine.

## 4. Messidor-2
- **Role**: **External Robustness and Domain-Shift Evaluation**
- **Function**: Messidor-2 will serve as a pure Out-Of-Distribution (OOD) test set. 
- **Constraint**: The official Messidor-2 distribution lacks ground-truth grading. Quantitative label use is strictly prohibited unless tied to a documented, legitimate, peer-reviewed annotation source (e.g., the standard Kaggle/Decathlon supplemental labels).
