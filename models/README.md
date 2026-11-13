# VisionX Model Weights

This directory holds trained PyTorch checkpoints for local inference and evaluation.

### Architecture
- **Stage 1 (Referral Gatekeeper)**: Binary classification separating Non-Referable (Grade 0-1) from Referable (Grade 2-4) DR with calibrated decision threshold $\tau = 0.30$.
- **Stage 2 (Specialist Stager)**: Multi-class severity classification (Moderate NPDR, Severe NPDR, Proliferative DR).
- **Grad-CAM**: Dynamic visual explanation heatmaps generated from the backbone convolutional feature maps.

### Checkpoints
Large binary model weight files (`*.pt`, `*.pth`) are excluded from Git version control per repository hygiene best practices.
In local development and automated testing, the Two-Stage pipeline (`ml.models.two_stage`) initializes with edge backbones (ResNet-18 / MobileNetV3 / EfficientNet-B0) and executes with zero external weight download required.

