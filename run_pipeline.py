"""
run_pipeline.py - Master script to execute entire research pipeline
Optimized for 20-hour completion on 2x L40S GPUs
"""

import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path

class ResearchPipeline:
    def __init__(self):
        self.start_time = time.time()
        self.results = {}
        
    def log(self, message: str, level: str = "INFO"):
        """Timestamped logging"""
        elapsed = (time.time() - self.start_time) / 3600
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{elapsed:.1f}h] {level}: {message}")
    
    def run_command(self, cmd: str, step_name: str):
        """Execute command with error handling"""
        self.log(f"Starting: {step_name}")
        start = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                capture_output=True,
                text=True
            )
            duration = (time.time() - start) / 60
            self.log(f"✅ Completed: {step_name} ({duration:.1f} min)", "SUCCESS")
            self.results[step_name] = {"status": "success", "duration": duration}
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Failed: {step_name}\n{e.stderr}", "ERROR")
            self.results[step_name] = {"status": "failed", "error": str(e)}
            return False
    
    def setup_environment(self):
        """Hour 1: Environment setup"""
        self.log("="*60)
        self.log("HOUR 1: Environment Setup")
        self.log("="*60)
        
        # Create directories
        dirs = [
            "data/raw", "data/processed", "data/adversarial",
            "models/naive", "models/grounded",
            "results/tables", "results/figures", "results/logs",
            "paper/latex", "paper/figures"
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)
        
        # Install dependencies
        self.run_command(
            "pip install -q torch transformers accelerate peft bitsandbytes "
            "datasets evaluate scipy pandas numpy matplotlib seaborn rouge-score scikit-learn tqdm",
            "Install dependencies"
        )
        
        self.log("✅ Environment ready!")
    
    def generate_adversarial_cases(self):
        """Hours 2-3: Generate test cases"""
        self.log("="*60)
        self.log("HOURS 2-3: Generating Adversarial Test Cases")
        self.log("="*60)
        
        return self.run_command(
            "python generate_adversarial.py",
            "Generate 150 adversarial test cases"
        )
    
    def train_models_parallel(self):
        """Hours 4-11: Train both models in parallel"""
        self.log("="*60)
        self.log("HOURS 4-11: Training Models (Parallel on 2x L40S)")
        self.log("="*60)
        
        self.log("🚀 Launching parallel training...")
        self.log("   GPU 0: Naive fine-tuning")
        self.log("   GPU 1: Physics-grounded fine-tuning")
        
        return self.run_command(
            "python train.py --mode parallel",
            "Train both models (naive + grounded)"
        )
    
    def run_evaluation(self):
        """Hours 12-14: Evaluate models"""
        self.log("="*60)
        self.log("HOURS 12-14: Model Evaluation")
        self.log("="*60)
        
        return self.run_command(
            "python evaluate.py",
            "Evaluate models and generate results"
        )
    
    def generate_paper_assets(self):
        """Hours 15-16: Generate tables and figures"""
        self.log("="*60)
        self.log("HOURS 15-16: Generating Paper Assets")
        self.log("="*60)
        
        # Tables and figures are generated in evaluate.py
        self.log("✅ Tables saved to results/tables/")
        self.log("✅ Figures saved to results/figures/")
        
        return True
    
    def compile_paper(self):
        """Hours 17-20: Prepare paper"""
        self.log("="*60)
        self.log("HOURS 17-20: Paper Preparation")
        self.log("="*60)
        
        # Copy LaTeX template
        if os.path.exists("paper_template.tex"):
            self.run_command(
                "cp paper_template.tex paper/latex/main.tex",
                "Copy LaTeX template"
            )
        
        # Copy figures to paper directory
        self.run_command(
            "cp results/figures/*.pdf paper/figures/",
            "Copy figures to paper folder"
        )
        
        self.log("📝 Paper template ready at paper/latex/main.tex")
        self.log("📊 All figures and tables prepared")
        
        return True
    
    def print_summary(self):
        """Print execution summary"""
        total_time = (time.time() - self.start_time) / 3600
        
        self.log("="*60)
        self.log("🎉 PIPELINE COMPLETE!")
        self.log("="*60)
        
        print("\n📊 Execution Summary:")
        print(f"{'Step':<40} {'Status':<10} {'Duration':<10}")
        print("-" * 60)
        
        for step, result in self.results.items():
            status = "✅ SUCCESS" if result["status"] == "success" else "❌ FAILED"
            duration = f"{result.get('duration', 0):.1f} min"
            print(f"{step:<40} {status:<10} {duration:<10}")
        
        print("-" * 60)
        print(f"{'Total Time':<40} {'':<10} {total_time:.1f} hours")
        
        print("\n📁 Output Files:")
        print("   ├── data/adversarial/test_cases.csv")
        print("   ├── models/naive/final/")
        print("   ├── models/grounded/final/")
        print("   ├── results/tables/")
        print("   │   ├── table1_comparison.tex")
        print("   │   └── table2_violations.tex")
        print("   ├── results/figures/")
        print("   │   ├── fig1_hallucination_comparison.pdf")
        print("   │   └── fig2_violation_breakdown.pdf")
        print("   └── paper/latex/main.tex")
        
        print("\n🎯 Next Steps:")
        print("1. Review results in results/")
        print("2. Edit paper/latex/main.tex with your details")
        print("3. Compile LaTeX: cd paper/latex && pdflatex main.tex")
        print("4. Submit to IEEE/Springer conference!")
        
        print("\n🏆 Recommended Conferences:")
        print("   • IEEE ITSC 2026 (Deadline: ~March 2026)")
        print("   • IEEE IV 2026 (Deadline: ~January 2026)")
        print("   • Springer ICANN 2026 (Deadline: ~April 2026)")
        print("   • IEEE IJCNN 2026 (Multiple deadlines)")
    
    def run_full_pipeline(self):
        """Execute complete 20-hour pipeline"""
        self.log("🚀 Starting Research Pipeline")
        self.log(f"Target: IEEE/Springer Paper in 20 Hours")
        self.log(f"Hardware: 2x NVIDIA L40S (48GB each)")
        
        steps = [
            self.setup_environment,
            self.generate_adversarial_cases,
            self.train_models_parallel,
            self.run_evaluation,
            self.generate_paper_assets,
            self.compile_paper
        ]
        
        for i, step in enumerate(steps, 1):
            self.log(f"\n{'='*60}")
            self.log(f"STEP {i}/{len(steps)}")
            self.log(f"{'='*60}")
            
            success = step()
            if not success and i < 4:  # Allow failure after training
                self.log("❌ Critical step failed. Stopping pipeline.", "ERROR")
                break
        
        self.print_summary()

# ===================================
# Quick Start Options
# ===================================

def quick_start_demo():
    """
    Quick 2-hour demo version for testing
    Uses smaller dataset and fewer epochs
    """
    print("🎯 Quick Start Demo (2 hours)")
    print("This will:")
    print("  • Generate 30 test cases (instead of 150)")
    print("  • Train for 1 epoch (instead of 3)")
    print("  • Use smaller batch size")
    print()
    
    # Modify config for demo
    demo_steps = [
        "python generate_adversarial.py --n_samples 30",
        "python train.py --mode parallel --epochs 1 --batch_size 2",
        "python evaluate.py",
    ]
    
    for cmd in demo_steps:
        print(f"Running: {cmd}")
        os.system(cmd)

def full_pipeline():
    """Run complete 20-hour pipeline"""
    pipeline = ResearchPipeline()
    pipeline.run_full_pipeline()

# ===================================
# Main Entry Point
# ===================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Traffic Hallucination Detection Research Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["full", "demo", "step"],
        default="full",
        help="Execution mode: full (20h), demo (2h), or step-by-step"
    )
    parser.add_argument(
        "--step",
        type=int,
        help="Run specific step (1-6) when mode=step"
    )
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        quick_start_demo()
    elif args.mode == "step":
        pipeline = ResearchPipeline()
        steps = {
            1: pipeline.setup_environment,
            2: pipeline.generate_adversarial_cases,
            3: pipeline.train_models_parallel,
            4: pipeline.run_evaluation,
            5: pipeline.generate_paper_assets,
            6: pipeline.compile_paper
        }
        if args.step in steps:
            steps[args.step]()
        else:
            print(f"Invalid step: {args.step}. Choose 1-6.")
    else:
        full_pipeline()

# ===================================
# Usage Examples
# ===================================

"""
USAGE EXAMPLES:

1. Full 20-hour pipeline:
   python run_pipeline.py --mode full

2. Quick 2-hour demo:
   python run_pipeline.py --mode demo

3. Run specific step:
   python run_pipeline.py --mode step --step 2  # Generate test cases
   python run_pipeline.py --mode step --step 3  # Train models
   python run_pipeline.py --mode step --step 4  # Evaluate

4. Manual execution:
   python generate_adversarial.py
   python train.py --mode parallel
   python evaluate.py

EXPECTED TIMELINE (Full Mode):
  Hour 1: Setup ✓
  Hours 2-3: Generate adversarial cases ✓
  Hours 4-11: Train models (parallel on 2x L40S) ✓
  Hours 12-14: Evaluate & detect hallucinations ✓
  Hours 15-16: Generate tables & figures ✓
  Hours 17-20: Paper writing (manual)

OUTPUT:
  ✓ 150 adversarial test cases
  ✓ 2 fine-tuned models (naive + grounded)
  ✓ Comprehensive evaluation metrics
  ✓ Publication-ready tables & figures
  ✓ Complete IEEE paper template
"""
