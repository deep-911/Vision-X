import torch
import time

def verify_cuda():
    print("--- PyTorch CUDA Verification ---")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Version (torch.version.cuda): {torch.version.cuda}")
    
    is_available = torch.cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if is_available:
        device_count = torch.cuda.device_count()
        print(f"Device Count: {device_count}")
        
        for i in range(device_count):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            print(f"Device {i}: {name} (Total Memory: {props.total_memory / (1024**2):.2f} MB)")
            
        print("\n--- Benchmark ---")
        N = 5000
        x = torch.rand(N, N).cuda()
        y = torch.rand(N, N).cuda()
        
        # Warmup
        for _ in range(5):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
            
        start = time.time()
        for _ in range(10):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"10x Matrix Multiplication ({N}x{N}) on GPU took: {end - start:.4f} seconds")
        print(f"Benchmark success! Output shape: {z.shape}, Device: {z.device}")
    else:
        print("CUDA is NOT available. Verification failed.")

if __name__ == "__main__":
    verify_cuda()
