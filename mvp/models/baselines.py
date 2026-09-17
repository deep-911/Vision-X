import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BaselineClassification(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class BaselineOrdinal(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        # CORAL style ordinal requires num_classes - 1 outputs
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes - 1)
        
    def forward(self, x):
        return self.backbone(x)
