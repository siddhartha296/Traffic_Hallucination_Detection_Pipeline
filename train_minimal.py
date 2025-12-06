"""
train_minimal.py - Memory-efficient training for 2x L40S
Reduces batch size and uses gradient checkpointing
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
from datasets import Dataset
import pandas as pd
import os
import argparse

def create_prompt(row, model_type):
    """Create training prompt"""
    if model_type == 'naive':
        return f"""Traffic Prediction Task:

Location: {row.get('road_segment', row.get('location', 'Unknown'))}
Timestamp: {row['timestamp']}
Observation: {row['explanation']}

Question: Is this traffic pattern physically possible and consistent? Explain your reasoning.

Analysis: {row['ground_truth']}"""
    else:  # grounded
        constraints = """Physical Constraints:
1. Max density: 100 veh/km
2. Max city speed: 60 km/h
3. Max highway speed: 130 km/h
4. Road capacity: 2000 veh/hour/lane
5. Traffic must have upstream cause
6. Patterns must match time-of-day"""
        
        return f"""Traffic Prediction with Physics Constraints:

{constraints}

Location: {row.get('road_segment', row.get('location', 'Unknown'))}
Timestamp: {row['timestamp']}
Observation: {row['explanation']}

Question: Verify if this violates any constraints. Check density, speed, temporal consistency, causality.

Analysis: {row['ground_truth']}"""

def train_model(model_type, gpu_id):
    """Train a single model"""
    
    device = f"cuda:{gpu_id}"
    print(f"\n🚀 Training {model_type} model on GPU {gpu_id}")
    print("="*60)
    
    # Load data
    print("📊 Loading training data...")
    df = pd.read_csv('data/adversarial/test_cases.csv')
    
    # Create prompts
    prompts = [create_prompt(row, model_type) for _, row in df.iterrows()]
    
    # Tokenizer
    print("📦 Loading tokenizer...")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Tokenize
    print("🔤 Tokenizing...")
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )
    
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone()
    })
    
    # Model with 4-bit quantization
    print(f"📦 Loading model on {device}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": gpu_id},
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments - REDUCED for memory
    output_dir = f"models/{model_type}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Reduced from 4
        gradient_accumulation_steps=8,  # Increased from 4
        learning_rate=2e-4,
        warmup_steps=50,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    print("🔥 Training started...")
    trainer.train()
    
    # Save
    print("💾 Saving model...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    
    print(f"✅ {model_type.capitalize()} model training complete!")
    print(f"   Saved to: {output_dir}/final")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["naive", "grounded", "both"], default="both")
    args = parser.parse_args()
    
    if args.model_type == "both":
        # Train both sequentially
        print("🚀 Training both models sequentially...")
        train_model("naive", gpu_id=0)
        
        # Clean up
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n⏳ Clearing GPU memory...")
        import time
        time.sleep(3)
        
        train_model("grounded", gpu_id=1)
    else:
        # Train single model
        gpu_id = 0 if args.model_type == "naive" else 1
        train_model(args.model_type, gpu_id)

if __name__ == "__main__":
    main()
