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
* **PyTorch Setup**: We are using the `cu130` (CUDA 13.0) build of PyTorch, avoiding PyPI fallbacks to ensure CUDA compatibility.
* **Versions**:
  - `torch==2.14.0+cu130`
  - `torchvision==0.29.0+cu130`
  - `torchaudio==2.11.0+cu130`
* **Installation Source**: `--index-url https://download.pytorch.org/whl/cu130`

## MATLAB/Simulink
* **Version**: R2026a (Version 26.1)
* **Toolboxes verified**: Simulink, Bioinformatics, Computer Vision, Deep Learning, Image Processing, Medical Imaging, Parallel Computing, SimEvents, Statistics and Machine Learning.
