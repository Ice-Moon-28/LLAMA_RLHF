# import os
# import gc
# import torch

# import transformers
# from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
# from datasets import load_dataset
# from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from trl import DPOTrainer, DPOConfig
# import bitsandbytes as bnb
# import wandb
# hf_token = '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'
# # # Defined in the secrets tab in Google Colab
# wb_token = '1c1fa66d79864363e5f33bb705a768da6cf094e5'
# # Defined in the secrets tab in Google Colab
# wandb.login(key=wb_token)
# from transformers import default_data_collator

# model_name = "teknium/OpenHermes-2.5-Mistral-7B"
# new_model = "DPO_NeuralHermes-2.5-Mistral-7B"
# cache_dir = "/root/autodl-tmp"

# def chatml_format(example):
#     # Format system
#     if len(example['system']) > 0:
#         message = {"role": "system", "content": example['system']}
#         system = tokenizer.apply_chat_template([message], tokenize=False)
#     else:
#         system = ""

#     # Format instruction
#     message = {"role": "user", "content": example['question']}
#     prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

#     # Format chosen answer
#     chosen = example['chosen'] + "<|im_end|>\n"

#     # Format rejected answer
#     rejected = example['rejected'] + "<|im_end|>\n"

#     return {
#         "prompt": system + prompt,
#         "chosen": chosen,
#         "rejected": rejected,
#     }

# # Load dataset
# dataset = load_dataset("Intel/orca_dpo_pairs")['train']

# # Save columns
# original_columns = dataset.column_names

# # Tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "left"

# # Format dataset
# dataset = dataset.map(
#     chatml_format,
#     remove_columns=original_columns
# )


# def print_grad_info(model):
#     for name, param in model.named_parameters():
#         if param.grad is not None:
#             print(f"Layer: {name}, Gradient dtype: {param.grad.dtype}, Gradient size: {param.grad.numel()}")
#         else:
#             print(f"Layer: {name}, Gradient: None")
# # LoRA configuration
# peft_config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
# )

# # Model to fine-tune
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     load_in_4bit=True,
#     cache_dir=cache_dir,
# )
# model.config.use_cache = False

# # 7B ===> 500 M 训练参数
# # Training arguments
# training_args = DPOConfig(
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=4,
#     gradient_checkpointing=True,
#     learning_rate=5e-5,
#     lr_scheduler_type="cosine",
#     max_steps=200,
#     # save_strategy="no",
#     logging_steps=1,
#     output_dir=new_model,
#     optim="paged_adamw_32bit",
#     warmup_steps=100,
#     bf16=True,
#     report_to="wandb",
#     beta=0.1,
#     max_prompt_length=1024,
#     max_length=1536,
#     log_level="debug",
# )

# # Create DPO trainer
# dpo_trainer = DPOTrainer(
#     model,
#     args=training_args,
#     train_dataset=dataset,
#     tokenizer=tokenizer,
#     peft_config=peft_config,
# )

# # Fine-tune model with DPO
# dpo_trainer.train()


from eval.pipeline import pipeline


if __name__ == "__main__":
    pipeline()