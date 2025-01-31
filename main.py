import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from torch.utils.data import ConcatDataset
import wandb

from dataset.get_dataset import get_antropic_dataset, get_intel_imdb_dataset, split_dataset
from dataset.get_weighted_dataset import ProbabilisticMultiDataset
from model.get_model import get_model
hf_token = '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'
# # Defined in the secrets tab in Google Colab
wb_token = '1c1fa66d79864363e5f33bb705a768da6cf094e5'
# Defined in the secrets tab in Google Colab
wandb.login(key=wb_token)
from transformers import default_data_collator

model_name = "teknium/OpenHermes-2.5-Mistral-7B"
new_model = "DPO_NeuralHermes-2.5-Mistral-7B_WITH_TWO_DATASET"
cache_dir = "/root/autodl-tmp"


def train():
    model, tokenizer = get_model(
        model_name=model_name,
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir" :cache_dir,
        },
    )

    intel_dataset = get_intel_imdb_dataset(tokenizer, 'train')

    antropic = get_antropic_dataset(tokenizer, 'train')

    dataset = ProbabilisticMultiDataset(
        datasets=[intel_dataset, antropic],
        sampling_probs=[0.7, 0.3]
    )

    dataset = split_dataset(dataset, 0.9, 3047)

    train_dataset, eval_dataset = dataset['train'], dataset['test']


    # LoRA configuration
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # 7B ===> 500 M 训练参数

    # Training arguments
    training_args = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        max_steps=500,
        # save_strategy="no",
        logging_steps=1,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        warmup_steps=50,
        evaluation_strategy="steps",  # 每隔一定步数进行 eval
        eval_steps=50,  # 例如每 50 步评估一次
        per_device_eval_batch_size=16,
        bf16=True,
        report_to="wandb",
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
        log_level="debug",
        seed=3047,
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )
    # Fine-tune model with DPO
    dpo_trainer.train()


if __name__ == "__main__":

    train()