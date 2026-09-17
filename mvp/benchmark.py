import time
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from pathlib import Path

def benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}...")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting benchmark.")
        return
        
    batch_sizes = [8, 16, 32]
    model = efficientnet_b0().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler('cuda')
    criterion = nn.CrossEntropyLoss()
    
    for bs in batch_sizes:
        try:
            print(f"\nTesting Batch Size {bs}...")
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
            
            # Dummy data
            imgs = torch.randn(bs, 3, 224, 224).to(device)
            labels = torch.randint(0, 5, (bs,)).to(device)
            
            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            # Time it
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(10):
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model(imgs)
                    loss = criterion(out, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            torch.cuda.synchronize()
            end = time.time()
            
            duration = (end - start) / 10.0
            img_per_sec = bs / duration
            mem = torch.cuda.max_memory_allocated(device) / (1024**2)
            
            print(f"  Duration per batch: {duration:.4f}s")
            print(f"  Throughput: {img_per_sec:.2f} images/s")
            print(f"  Peak VRAM: {mem:.2f} MB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at Batch Size {bs}!")
                torch.cuda.empty_cache()
            else:
                raise e

if __name__ == "__main__":
    benchmark()
