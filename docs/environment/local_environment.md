# Local Environment Specification

## Hardware
* **GPU**: NVIDIA GeForce RTX 3050 Laptop GPU
* **VRAM**: 4096 MiB (4 GB)
* **Driver Version**: 616.92
* **CUDA UMD Version**: 13.4

## Software Stack
* **Python Version**: 3.14.6
* **Virtual Environment**: `.venv/` (Excluded from git)
* **Dependency Manager**: pip 26.1.2

## Machine Learning Framework
* **PyTorch Setup**: We are using the `cu124` (CUDA 12.4) build of PyTorch, which is compatible with the local driver (616.92) and CUDA 13.4 backend.
* **Installation Source**: `https://download.pytorch.org/whl/cu124`

## MATLAB/Simulink
* **Version**: R2026a (Version 26.1)
* **Toolboxes verified**: Simulink, Bioinformatics, Computer Vision, Deep Learning, Image Processing, Medical Imaging, Parallel Computing, SimEvents, Statistics and Machine Learning.
