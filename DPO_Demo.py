import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel, LoraConfig
from trl import DPOTrainer, DPOConfig


model_name = 'Qwen/Qwen2-0.5B-Instruct'
cache_dir = "Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")



# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
)

# Model to fine-tune
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    cache_dir=cache_dir,
)
model.config.use_cache = False

# Training arguments
training_args = DPOConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    learning_rate=5e-5,
    lr_scheduler_type="cosine",
    max_steps=200,
    save_strategy="no",
    logging_steps=1,
    output_dir=new_model,
    optim="paged_adamw_32bit",
    warmup_steps=100,
    bf16=True,
    report_to="wandb",
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

# Create DPO trainer
dpo_trainer = DPOTrainer(
    model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config,
)

# Fine-tune model with DPO
dpo_trainer.train()