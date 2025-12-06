"""
train.py - Parallel training for naive and physics-grounded models
Optimized for 2x L40S GPUs (48GB each)
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset, load_dataset
import pandas as pd
import numpy as np
from typing import Dict, List
import json
import argparse
import os

class TrafficPredictionTrainer:
    def __init__(self, model_type: str, gpu_id: int):
        """
        model_type: 'naive' or 'grounded'
        gpu_id: 0 or 1 for multi-GPU setup
        """
        self.model_type = model_type
        self.device = f"cuda:{gpu_id}"
        self.gpu_id = gpu_id
        
        print(f"🚀 Initializing {model_type} trainer on GPU {gpu_id}")
        
        # Model configuration
        self.model_name = "mistralai/Mistral-7B-v0.1"
        self.max_length = 512
        
        # 4-bit quantization for memory efficiency
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
    def load_model_and_tokenizer(self):
        """Load model with quantization and LoRA"""
        print(f"📦 Loading {self.model_name} on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load model with 4-bit quantization
        # Fixed: Proper device mapping for multiprocessing
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map={"": self.device},
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)
        
        # Add LoRA adapters
        self.model = get_peft_model(model, self.lora_config)
        
        print(f"✅ Model loaded. Trainable params: {self.model.print_trainable_parameters()}")
        
    def create_training_data(self) -> Dataset:
        """Create training dataset with appropriate prompting"""
        print("📊 Creating training dataset...")
        
        # Load adversarial test cases
        adversarial_df = pd.read_csv('data/adversarial/test_cases.csv')
        
        # Create prompts based on model type
        training_examples = []
        
        for _, row in adversarial_df.iterrows():
            if self.model_type == 'naive':
                # Naive: Just predict without constraints
                prompt = self._create_naive_prompt(row)
            else:  # grounded
                # Grounded: Include physics constraints
                prompt = self._create_grounded_prompt(row)
            
            training_examples.append({
                'input': prompt,
                'output': row['ground_truth'],
                'is_hallucination': row['hallucination']
            })
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_pandas(pd.DataFrame(training_examples))
        
        # Tokenize
        def tokenize_function(examples):
            inputs = examples['input']
            targets = examples['output']
            
            # Combine input and output for causal LM
            full_text = [f"{inp}\n\nAnalysis: {tgt}" for inp, tgt in zip(inputs, targets)]
            
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length"
            )
            
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        print(f"✅ Created {len(tokenized_dataset)} training examples")
        return tokenized_dataset
    
    def _create_naive_prompt(self, row: pd.Series) -> str:
        """Create prompt without physics constraints"""
        return f"""Traffic Prediction Task:

Location: {row.get('road_segment', row.get('location', 'Unknown'))}
Timestamp: {row['timestamp']}
Observation: {row['explanation']}

Question: Is this traffic pattern physically possible and consistent? Explain your reasoning."""
    
    def _create_grounded_prompt(self, row: pd.Series) -> str:
        """Create prompt with physics constraints"""
        constraints = """
Physical Constraints:
1. Maximum density: 100 vehicles/km
2. Maximum city speed: 60 km/h
3. Maximum highway speed: 130 km/h
4. Road capacity: 2000 vehicles/hour/lane
5. Traffic must have upstream cause
6. Temporal patterns must match time-of-day
"""
        
        return f"""Traffic Prediction Task with Physics Constraints:

{constraints}

Location: {row.get('road_segment', row.get('location', 'Unknown'))}
Timestamp: {row['timestamp']}
Observation: {row['explanation']}

Question: Verify if this observation violates any physical constraints. Check:
- Density limits
- Speed limits
- Temporal consistency
- Causality

Provide detailed analysis."""
    
    def train(self):
        """Train the model"""
        print(f"🏋️ Starting training for {self.model_type} model...")
        
        # Load model
        self.load_model_and_tokenizer()
        
        # Create dataset
        train_dataset = self.create_training_data()
        
        # Training arguments
        output_dir = f"models/{self.model_type}"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            warmup_steps=100,
            logging_steps=50,
            save_steps=500,
            save_total_limit=2,
            fp16=True,
            report_to="none",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
            dataloader_pin_memory=False,
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        # Train
        print("🔥 Training started...")
        trainer.train()
        
        # Save final model
        trainer.save_model(f"{output_dir}/final")
        self.tokenizer.save_pretrained(f"{output_dir}/final")
        
        print(f"✅ Training complete! Model saved to {output_dir}/final")

def train_parallel_models():
    """Launch training for both models in parallel"""
    import multiprocessing as mp
    
    def train_model(model_type, gpu_id):
        trainer = TrafficPredictionTrainer(model_type, gpu_id)
        trainer.train()
    
    # Start both training processes
    processes = []
    
    # GPU 0: Naive model
    p1 = mp.Process(target=train_model, args=("naive", 0))
    p1.start()
    processes.append(p1)
    
    # GPU 1: Grounded model
    p2 = mp.Process(target=train_model, args=("grounded", 1))
    p2.start()
    processes.append(p2)
    
    # Wait for both to complete
    for p in processes:
        p.join()
    
    print("✅ Both models trained successfully!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["parallel", "single"], default="parallel")
    parser.add_argument("--model_type", choices=["naive", "grounded"], default="naive")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.mode == "parallel":
        print("🚀 Launching parallel training on 2x L40S...")
        train_parallel_models()
    else:
        trainer = TrafficPredictionTrainer(args.model_type, args.gpu_id)
        trainer.train()

if __name__ == "__main__":
    main()