import torch
from mvp.models.baselines import build_baseline
import yaml
import sys

def main():
    print("Running PyTorch Smoke Test on tensor of ones...")
    with open("mvp/configs/baseline_softmax.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    model = build_baseline(config['model'])
    model.load_state_dict(torch.load("artifacts/checkpoints/mvp_baseline_softmax/best.pt", map_location='cpu'))
    model.eval()
    
    # Input is ones (batch_size=1, C=3, H=224, W=224)
    dummy_input = torch.ones((1, 3, 224, 224), dtype=torch.float32)
    
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.softmax(logits, dim=1)
        
    print("PyTorch Predictions (Probabilities):")
    print(probs.numpy().flatten().tolist())

if __name__ == "__main__":
    main()
