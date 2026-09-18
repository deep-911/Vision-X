import torch.nn as nn
import segmentation_models_pytorch as smp

def get_unet_model(encoder_name="efficientnet-b0", num_classes=4):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model
