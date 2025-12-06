#!/bin/bash

echo "🚀 Setting up Traffic Hallucination Detection Pipeline"
echo "========================================================"

# Create project structure
echo "📁 Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/adversarial
mkdir -p models/naive
mkdir -p models/grounded
mkdir -p results/tables
mkdir -p results/figures
mkdir -p results/logs
mkdir -p paper/latex
mkdir -p paper/figures

echo "✅ Directories created!"

# Check Python version
echo ""
echo "🐍 Checking Python version..."
python --version

# Check for GPU
echo ""
echo "🎮 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "⚠️  Warning: nvidia-smi not found"

# Install dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install --user torch>=2.1.0 transformers>=4.35.0 accelerate>=0.25.0 \
    peft>=0.7.0 bitsandbytes>=0.41.0 datasets>=2.14.0 evaluate>=0.4.0 \
    scipy>=1.11.0 pandas>=2.0.0 numpy>=1.24.0 matplotlib>=3.7.0 \
    seaborn>=0.12.0 rouge-score>=0.1.2 scikit-learn>=1.3.0 tqdm>=4.65.0 \
    h5py>=3.9.0 pillow>=10.0.0

echo ""
echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   1. Run: python generate_adversarial.py"
echo "   2. Run: python train.py --mode parallel"
echo "   3. Run: python evaluate.py"
echo ""
echo "Or run the full pipeline: python run_pipeline.py --mode full"
