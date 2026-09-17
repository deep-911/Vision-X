import torch
import torch.nn as nn
import torch.nn.functional as F

def coral_encoding(labels, num_classes):
    """
    Convert discrete integer labels to CORAL rank-consistent binary targets.
    Example for num_classes=5:
      y=0 -> [0, 0, 0, 0]
      y=1 -> [1, 0, 0, 0]
      y=2 -> [1, 1, 0, 0]
      y=3 -> [1, 1, 1, 0]
      y=4 -> [1, 1, 1, 1]
    """
    levels = torch.arange(num_classes - 1, device=labels.device)
    # levels: [0, 1, 2, 3]
    # labels.unsqueeze(1): [[y1], [y2], ...]
    # target > levels returns True/False, we convert to float
    encoded = (labels.unsqueeze(1) > levels).float()
    return encoded

def coral_decoding(logits):
    """
    Convert CORAL cumulative logits back to predicted class indices.
    Logits are shape (batch_size, num_classes-1).
    Probas = sigmoid(logits).
    For each example, predicted rank is sum(probas > 0.5).
    """
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_classes = torch.sum(predict_levels, dim=1)
    return predicted_classes

class CORALLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, logits, labels):
        """
        logits: (batch_size, num_classes - 1)
        labels: (batch_size,) integer labels
        """
        num_classes = logits.shape[1] + 1
        targets = coral_encoding(labels, num_classes)
        # We use standard binary cross entropy on each of the K-1 logits
        return F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
