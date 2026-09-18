import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, validities, smooth=1):
        # inputs are raw logits
        # targets are binary masks (0 or 1)
        # validities are [B, C] representing whether the mask for that class is valid (1) or missing (0)
        
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        # shape [B, C, H, W] -> [B, C, H*W]
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        inputs_sigmoid_flat = inputs_sigmoid.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none') # [B, C, H*W]
        bce_loss = bce_loss.mean(dim=2) # [B, C]
        
        intersection = (inputs_sigmoid_flat * targets_flat).sum(dim=2) # [B, C]
        dice = (2.*intersection + smooth)/(inputs_sigmoid_flat.sum(dim=2) + targets_flat.sum(dim=2) + smooth)
        dice_loss = 1 - dice # [B, C]
        
        total_loss = bce_loss + dice_loss # [B, C]
        
        # Apply validity mask
        # validities shape is [B, C]
        total_loss = total_loss * validities
        
        # Mean only over valid samples
        valid_count = validities.sum()
        if valid_count > 0:
            return total_loss.sum() / valid_count
        else:
            return total_loss.sum() * 0.0 # return 0 with grad

class WeightedFocalDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, class_weights=None):
        super(WeightedFocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights # list or tensor of length C

    def forward(self, inputs, targets, validities, smooth=1):
        # inputs: [B, C, H, W] logits
        # targets: [B, C, H, W] binary
        
        inputs_sigmoid = torch.sigmoid(inputs)
        
        inputs_flat = inputs.view(inputs.size(0), inputs.size(1), -1)
        inputs_sigmoid_flat = inputs_sigmoid.view(inputs.size(0), inputs.size(1), -1)
        targets_flat = targets.view(targets.size(0), targets.size(1), -1)
        
        # Standard BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs_flat, targets_flat, reduction='none') # [B, C, H*W]
        
        # Focal scaling: p_t is the probability of the true class
        # p_t = p if y==1 else (1-p)
        p_t = inputs_sigmoid_flat * targets_flat + (1 - inputs_sigmoid_flat) * (1 - targets_flat)
        focal_weight = (1 - p_t) ** self.gamma
        
        # alpha balancing (optional, usually applied to positive class)
        # alpha_t = alpha if y==1 else (1-alpha)
        if self.alpha is not None:
            alpha_t = self.alpha * targets_flat + (1 - self.alpha) * (1 - targets_flat)
            focal_weight = alpha_t * focal_weight
            
        focal_loss = focal_weight * bce_loss
        focal_loss = focal_loss.mean(dim=2) # [B, C]
        
        # Dice
        intersection = (inputs_sigmoid_flat * targets_flat).sum(dim=2)
        dice = (2. * intersection + smooth) / (inputs_sigmoid_flat.sum(dim=2) + targets_flat.sum(dim=2) + smooth)
        dice_loss = 1 - dice # [B, C]
        
        # Total per-class loss before validity and class weights
        total_loss = focal_loss + dice_loss # [B, C]
        
        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, device=total_loss.device).view(1, -1)
            total_loss = total_loss * weights
            
        # Apply validity mask
        total_loss = total_loss * validities
        
        # Mean only over valid samples
        valid_count = validities.sum()
        if valid_count > 0:
            return total_loss.sum() / valid_count
        else:
            return total_loss.sum() * 0.0

