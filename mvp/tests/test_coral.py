import sys
import torch
import torch.nn as nn
from pathlib import Path

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.training.loss import coral_encoding, coral_decoding, CORALLoss
from mvp.models.baselines import BaselineOrdinal, CoralLayer

def test_coral_encoding():
    labels = torch.tensor([0, 1, 2, 3, 4])
    encoded = coral_encoding(labels, num_classes=5)
    
    expected = torch.tensor([
        [0, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]
    ], dtype=torch.float32)
    
    assert torch.allclose(encoded, expected), f"Encoding mismatch: {encoded} != {expected}"
    print("test_coral_encoding passed!")

def test_coral_decoding():
    # Simulated logits where sigmoid(logits) gives probabilities
    # Let's say logits are highly positive (>0) for 1s, highly negative (<0) for 0s
    logits = torch.tensor([
        [-10., -10., -10., -10.], # -> probas ≈ 0, sum = 0
        [ 10., -10., -10., -10.], # -> probas ≈ [1, 0, 0, 0], sum = 1
        [ 10.,  10., -10., -10.], # -> probas ≈ [1, 1, 0, 0], sum = 2
        [ 10.,  10.,  10., -10.], # -> probas ≈ [1, 1, 1, 0], sum = 3
        [ 10.,  10.,  10.,  10.]  # -> probas ≈ [1, 1, 1, 1], sum = 4
    ])
    
    predicted = coral_decoding(logits)
    expected = torch.tensor([0, 1, 2, 3, 4])
    
    assert torch.equal(predicted, expected), f"Decoding mismatch: {predicted} != {expected}"
    print("test_coral_decoding passed!")

def test_coral_layer_shape():
    layer = CoralLayer(in_features=128, num_classes=5)
    x = torch.randn(4, 128)
    out = layer(x)
    assert out.shape == (4, 4), f"Shape mismatch: {out.shape} != (4, 4)"
    
    # Test rank-consistency approximation implicitly:
    # Because biases are independent and weights are shared, 
    # if biases are sorted monotonically, probabilities will be monotonic.
    print("test_coral_layer_shape passed!")

def test_coral_loss():
    loss_fn = CORALLoss()
    logits = torch.randn(4, 4)
    labels = torch.tensor([0, 2, 4, 1])
    loss = loss_fn(logits, labels)
    assert loss.dim() == 0 and loss.item() >= 0, "Loss calculation failed or returned invalid shape"
    print("test_coral_loss passed!")

if __name__ == "__main__":
    test_coral_encoding()
    test_coral_decoding()
    test_coral_layer_shape()
    test_coral_loss()
    print("All CORAL tests passed!")
