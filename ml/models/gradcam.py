"""
VisionX: Explainable AI - Grad-CAM Visual Attention Heatmap Generator
======================================================================
Problem Statement: SIH26038 (MathWorks & Telemedicine Track)
Team: Indian_Gryffindor

Generates clinically interpretable Gradient-weighted Class Activation Mapping (Grad-CAM)
heatmaps over retinal fundus images for rural Primary Health Centre (PHC) edge deployment.
Visualizes why the model flagged referable pathology (e.g., hemorrhages, exudates,
neovascularization) to aid ophthalmologists in quick (<30s) verification.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation for convolutional neural network backbones
    (e.g., ResNet-18, MobileNetV3, EfficientNet-B0).
    
    Attaches forward and backward hooks to the last convolutional layer to compute
    the gradient of the target class score with respect to feature activation maps.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: PyTorch model in eval mode.
            target_layer: The target convolutional layer. If None, auto-detects
                          the last Conv2d layer in the network.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            target_layer = self._find_last_conv_layer(model)
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register_hooks()

    def _find_last_conv_layer(self, model: nn.Module) -> nn.Module:
        """Finds the last Conv2d module in the network architecture."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is None:
            raise ValueError("No Conv2d layer found in model for Grad-CAM hooks.")
        return last_conv

    def _register_hooks(self):
        """Registers forward and backward hooks on the target layer."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self._handles.extend([h_fwd, h_bwd])

    def remove_hooks(self):
        """Removes registered hooks to prevent memory leaks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generates a 2D Grad-CAM heatmap for a single input image.

        Args:
            input_tensor: [1, 3, H, W] normalized float tensor.
            target_class: Class index to explain. If None, uses argmax prediction.
            target_size: (width, height) to resize output heatmap to. If None,
                         matches input_tensor spatial dimensions.

        Returns:
            np.ndarray: 2D float array in range [0.0, 1.0] representing normalized attention.
        """
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad_(True)

        self.model.zero_grad()
        logits = self.model(input_tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if hooks were not invoked
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.zeros((h, w), dtype=np.float32)

        # Global average pooling of gradients: alpha_k = (1/Z) * sum(gradients)
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # [1, C, 1, 1]

        # Weighted combination of activation maps: L_c = ReLU(sum(alpha_k * A_k))
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # [1, 1, H', W']
        cam = F.relu(cam)

        cam_np = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max > cam_min:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)

        # Resize to target size
        if target_size is not None:
            out_w, out_h = target_size
        else:
            out_h, out_w = input_tensor.shape[2], input_tensor.shape[3]

        heatmap_resized = cv2.resize(cam_np, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def overlay_heatmap(
        image_rgb: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.40,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Blends a Grad-CAM heatmap over an RGB fundus image.

        Args:
            image_rgb: [H, W, 3] uint8 RGB image.
            heatmap: [H, W] float array in [0.0, 1.0].
            alpha: Heatmap blend opacity factor (0.0 to 1.0).
            colormap: OpenCV colormap (default: COLORMAP_JET).

        Returns:
            np.ndarray: [H, W, 3] uint8 RGB image with blended attention heatmap.
        """
        h, w = image_rgb.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        heatmap_uint8 = (heatmap * 255.0).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
        heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        blended = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap_colored_rgb, alpha, 0)
        return blended
