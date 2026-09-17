# VisionX - SIH26038 Project Status

## Hardware Environment (Local)
* GPU: NVIDIA GeForce RTX 3050 Laptop GPU
* VRAM: 4096 MiB (4 GB)
* Compute Environment: Python with PyTorch 2.14.0+cpu (Note: PyTorch CUDA needs to be installed, currently reporting `torch.cuda.is_available() == False`).
* MATLAB: Version 26.1 (R2026a) with Simulink, Deep Learning Toolbox, Image Processing Toolbox, Medical Imaging Toolbox, and SimEvents.

## Software Architecture Context
* Local Environment: Used for development, MVP implementations, preprocessing, and edge deployment simulation via MATLAB/Simulink.
* Remote Environment: Google Colab is designated for heavy training tasks, keeping models reproducible and hardware-independent.

## Datasets Status
* **IDRiD**: Present (`A. Segmentation.zip`, `B. Disease Grading.zip`, `C. Localization.zip`).
* **APTOS**: Missing.
* **EyePACS Kaggle DR**: Missing.
* **Messidor-2**: Missing.
