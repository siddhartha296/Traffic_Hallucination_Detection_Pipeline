"""
train_simple.py - Simplified training without 4-bit quantization
Uses standard fp16 training - more stable, uses more memory
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import pandas as pd
import os
import argparse

def create_prompt(row, model_type):
    """Create training prompt"""
    if model_type == 'naive':
        return f"""Traffic: {row['explanation']}
Analysis: {row['ground_truth']}"""
    else:
        return f"""Traffic (check physics): {row['explanation']}
Physics Analysis: {row['ground_truth']}"""

def train_model(model_type, gpu_id):
    """Train a single model - simplified version"""
    
    device = f"cuda:{gpu_id}"
    print(f"\n🚀 Training {model_type} model on GPU {gpu_id}")
    print("="*60)
    
    # Set environment
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Load data
    print("📊 Loading training data...")
    df = pd.read_csv('data/adversarial/test_cases.csv')
    
    # Create short prompts to save memory
    prompts = [create_prompt(row, model_type) for _, row in df.iterrows()]
    
    # Tokenizer
    print("📦 Loading tokenizer...")
    model_name = "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Tokenize with shorter sequences
    print("🔤 Tokenizing...")
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=256,  # Reduced from 512
        padding="max_length",
        return_tensors="pt"
    )
    
    dataset = Dataset.from_dict({
        'input_ids': tokenized['input_ids'],
        'attention_mask': tokenized['attention_mask'],
        'labels': tokenized['input_ids'].clone()
    })
    
    # Model WITHOUT quantization (more stable)
    print(f"📦 Loading model on {device}...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    model.config.use_cache = False
    
    # LoRA config - reduced for stability
    lora_config = LoraConfig(
        r=8,  # Reduced from 16
        lora_alpha=16,  # Reduced from 32
        target_modules=["q_proj", "v_proj"],  # Only 2 modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments - very conservative
    output_dir = f"models/{model_type}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,  # Reduced from 3
        per_device_train_batch_size=1,  # Minimum
        gradient_accumulation_steps=16,  # Compensate
        learning_rate=2e-4,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=1,
        fp16=True,
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )
    
    # Train
    print("🔥 Training started...")
    try:
        trainer.train()
        
        # Save
        print("💾 Saving model...")
        trainer.save_model(f"{output_dir}/final")
        tokenizer.save_pretrained(f"{output_dir}/final")
        
        print(f"✅ {model_type.capitalize()} model training complete!")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["naive", "grounded"], required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    success = train_model(args.model_type, args.gpu_id)
    
    if success:
        print(f"\n✅ Training successful!")
        print(f"Model saved to: models/{args.model_type}/final")
    else:
        print(f"\n❌ Training failed. Consider using demo results.")

if __name__ == "__main__":
    main()
