import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegformerMultiLabel(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super(SegformerMultiLabel, self).__init__()
        # mit-b2 backbone
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b2-finetuned-ade-512-512",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # Upsample logits to match input resolution (SegFormer outputs 1/4th resolution)
        logits = nn.functional.interpolate(
            outputs.logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        return logits
