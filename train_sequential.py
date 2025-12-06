"""
train_sequential.py - Sequential training (avoids multiprocessing issues)
Train one model at a time on each GPU
"""

import torch
import os
import gc
from train import TrafficPredictionTrainer

def train_sequential():
    """Train models sequentially to avoid multiprocessing issues"""
    
    print("="*60)
    print("🚀 Sequential Training Pipeline")
    print("Training models one at a time to avoid CUDA issues")
    print("="*60)
    
    # Train naive model on GPU 0
    print("\n" + "="*60)
    print("STEP 1/2: Training Naive Model on GPU 0")
    print("="*60)
    
    try:
        naive_trainer = TrafficPredictionTrainer("naive", gpu_id=0)
        naive_trainer.train()
        print("✅ Naive model training complete!")
    except Exception as e:
        print(f"❌ Naive model training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up GPU memory
    del naive_trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n⏳ Waiting for GPU 0 to clear...")
    import time
    time.sleep(5)
    
    # Train grounded model on GPU 1
    print("\n" + "="*60)
    print("STEP 2/2: Training Grounded Model on GPU 1")
    print("="*60)
    
    try:
        grounded_trainer = TrafficPredictionTrainer("grounded", gpu_id=1)
        grounded_trainer.train()
        print("✅ Grounded model training complete!")
    except Exception as e:
        print(f"❌ Grounded model training failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up
    del grounded_trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*60)
    print("🎉 All Training Complete!")
    print("="*60)
    print("Models saved to:")
    print("  - models/naive/final/")
    print("  - models/grounded/final/")

if __name__ == "__main__":
    train_sequential()
