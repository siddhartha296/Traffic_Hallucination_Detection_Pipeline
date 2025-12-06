"""
test_setup.py - Quick test to verify everything works
Run this before the full pipeline
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🧪 Testing imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('peft', 'PEFT'),
        ('datasets', 'Datasets'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
    ]
    
    failed = []
    for package, name in packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name}")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Missing packages: {', '.join(failed)}")
        print("Install with: pip install --user " + " ".join(failed))
        return False
    
    print("\n✅ All imports successful!")
    return True

def test_gpu():
    """Test GPU availability"""
    print("\n🎮 Testing GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  ✅ Found {gpu_count} GPU(s)")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"     GPU {i}: {name} ({mem:.1f} GB)")
            return True
        else:
            print("  ❌ No GPU found!")
            print("  ⚠️  Training will be very slow on CPU")
            return False
    except Exception as e:
        print(f"  ❌ Error checking GPU: {e}")
        return False

def test_directories():
    """Test directory structure"""
    print("\n📁 Testing directories...")
    
    required_dirs = [
        'data/adversarial',
        'models/naive',
        'models/grounded',
        'results/tables',
        'results/figures',
    ]
    
    missing = []
    for d in required_dirs:
        if os.path.exists(d):
            print(f"  ✅ {d}")
        else:
            print(f"  ⚠️  {d} (will create)")
            os.makedirs(d, exist_ok=True)
    
    print("\n✅ All directories ready!")
    return True

def test_adversarial_generation():
    """Test adversarial case generation"""
    print("\n🧪 Testing adversarial generation...")
    
    try:
        from generate_adversarial import AdversarialTrafficGenerator
        
        generator = AdversarialTrafficGenerator()
        
        # Test physics violations
        physics = generator.generate_physics_violations(5)
        print(f"  ✅ Physics violations: {len(physics)} cases")
        
        # Test temporal inconsistencies
        temporal = generator.generate_temporal_inconsistencies(5)
        print(f"  ✅ Temporal inconsistencies: {len(temporal)} cases")
        
        # Test spatial anomalies
        spatial = generator.generate_spatial_anomalies(5)
        print(f"  ✅ Spatial anomalies: {len(spatial)} cases")
        
        print("\n✅ Adversarial generation working!")
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("🚀 Traffic Hallucination Detection - Setup Test")
    print("="*60)
    
    results = {
        'imports': test_imports(),
        'gpu': test_gpu(),
        'directories': test_directories(),
        'adversarial': test_adversarial_generation(),
    }
    
    print("\n" + "="*60)
    print("📊 Test Summary")
    print("="*60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test.capitalize():<20} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 All tests passed! Ready to run pipeline.")
        print("\nNext steps:")
        print("  1. python generate_adversarial.py")
        print("  2. python train.py --mode parallel")
        print("  3. python evaluate.py")
        print("\nOr run full pipeline: python run_pipeline.py --mode full")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
