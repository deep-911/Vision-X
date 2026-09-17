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

class CoralLayer(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        # Shared weight vector: (in_features, 1)
        self.weight = nn.Parameter(torch.empty(in_features, 1))
        # Independent biases for each rank: (num_classes - 1)
        self.bias = nn.Parameter(torch.empty(num_classes - 1))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=torch.math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / torch.math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # x is (batch_size, in_features)
        # out is (batch_size, 1) + (num_classes - 1) = (batch_size, num_classes - 1)
        return torch.matmul(x, self.weight) + self.bias

class BaselineOrdinal(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        # CORAL style ordinal requires shared weights and independent biases
        self.backbone.classifier[1] = CoralLayer(in_features, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
