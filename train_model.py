"""Fine-tune model on your writing style."""
import json
import sqlite3
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch

print("🚀 Starting fine-tuning process...")

# Load conversations
print("📚 Loading your conversations...")
conn = sqlite3.connect('data/conversations.db')
query = "SELECT full_text FROM conversations WHERE length(full_text) > 500"
texts = [row[0] for row in conn.execute(query).fetchall()]
conn.close()

print(f"✓ Loaded {len(texts)} conversations")
print(f"✓ Total words: {sum(len(t.split()) for t in texts):,}")

# Prepare dataset
dataset = Dataset.from_dict({"text": texts})
print(f"✓ Dataset prepared")

# Load base model
print("🤖 Loading base model...")
model_name = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✓ Model loaded")

# Configure LoRA
print("⚙️ Configuring LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
print("✓ LoRA configured")
model.print_trainable_parameters()

# Tokenize dataset
print("🔤 Tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset.column_names
)
print("✓ Dataset tokenized")

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training configuration
print("🎯 Setting up training...")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    warmup_steps=100,
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Train
print("🏋️ Training starting...")
print("This takes 2-4 hours. Loss should decrease from ~3.0 to ~1.5")
trainer.train()

# Save
print("💾 Saving fine-tuned model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("✅ Training complete!")
print("Model saved to: ./fine_tuned_model")
