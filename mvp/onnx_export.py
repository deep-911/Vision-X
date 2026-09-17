import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from mvp.models.baselines import BaselineClassification, BaselineOrdinal

def export_onnx(config_path, checkpoint_path, output_path):
    print(f"Exporting model from {checkpoint_path} to ONNX format...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cpu")
    model_type = config['model']['type']
    num_classes = config['model']['num_classes']
    img_size = tuple(config['dataset']['img_size'])
    
    if model_type == 'classification':
        model = BaselineClassification(num_classes=num_classes)
    elif model_type == 'ordinal':
        model = BaselineOrdinal(num_classes=num_classes)
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()
    
    # Dummy input matching the expected image size
    dummy_input = torch.randn(1, 3, img_size[0], img_size[1])
    
    torch.onnx.export(
        model, 
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=14, 
        do_constant_folding=True, 
        input_names=['input'], 
        output_names=['output'], 
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"Successfully exported to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="model.onnx")
    args = parser.parse_args()
    
    export_onnx(args.config, args.checkpoint, args.output)
