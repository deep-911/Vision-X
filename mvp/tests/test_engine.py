import sys
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import os

base_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(base_dir))

from mvp.models.baselines import BaselineClassification, BaselineOrdinal
from mvp.training.loss import CORALLoss

def run_tiny_test(model_type='classification'):
    print(f"Running tiny {model_type} test...")
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'classification':
        model = BaselineClassification(num_classes=5).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = BaselineOrdinal(num_classes=5).to(device)
        criterion = CORALLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Dummy data
    imgs = torch.randn(4, 3, 224, 224).to(device)
    labels = torch.tensor([0, 2, 4, 1]).to(device)
    
    model.train()
    
    # Overfit a single batch for 5 steps to verify gradients
    initial_loss = None
    for step in range(5):
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        if step == 0:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()
        
    final_loss = loss.item()
    
    print(f"Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
    assert final_loss < initial_loss, f"Loss did not decrease for {model_type}"
    
    # Checkpoint save/load
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test.pt")
        torch.save(model.state_dict(), ckpt_path)
        
        # Load
        if model_type == 'classification':
            model_new = BaselineClassification(num_classes=5).to(device)
        else:
            model_new = BaselineOrdinal(num_classes=5).to(device)
            
        model_new.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model_new.eval()
        
        # Verify deterministic output
        with torch.no_grad():
            logits_loaded = model_new(imgs)
            # Since model is in eval(), recompute deterministic logits from the trained model to compare
            model.eval()
            logits_trained = model(imgs)
            assert torch.allclose(logits_trained, logits_loaded, atol=1e-5), "Checkpoint loading altered predictions"
            
    print(f"{model_type} tiny test passed!\n")

def test_engine_all():
    run_tiny_test('classification')
    run_tiny_test('ordinal')

if __name__ == "__main__":
    test_engine_all()
