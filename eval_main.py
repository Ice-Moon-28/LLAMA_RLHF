import random
import numpy as np
import torch
import wandb

from dataset.get_dataset import get_antropic_dataset, get_intel_imdb_dataset, split_dataset
from eval.pipeline import evaluate
from model.get_model import get_model
from peft import LoraConfig, PeftConfig, PeftModel
from trl import DPOTrainer, DPOConfig
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM

hf_token = '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'
# # Defined in the secrets tab in Google Colab
wb_token = '1c1fa66d79864363e5f33bb705a768da6cf094e5'
# Defined in the secrets tab in Google Colab
wandb.login(key=wb_token)
from transformers import default_data_collator

model_name = "teknium/OpenHermes-2.5-Mistral-7B"
new_model = "DPO_NeuralHermes-2.5-Mistral-7B/eval"
cache_dir = "/root/autodl-tmp"



def evaluate_trained(new_model):
    # 1. 加载基础模型 & Tokenizer
    model, tokenizer = get_model(
        model_name=model_name,
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
        },
    )

    # 3. 使用 `model` 直接加载 PeftModel
    model = PeftModel.from_pretrained(model, new_model)

    ref_model, _ = get_model(
        model_name="teknium/OpenHermes-2.5-Mistral-7B",
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
        },
    )

    dataset = get_antropic_dataset(tokenizer, 'train')

    dataset = split_dataset(dataset, 0.99, 3047)

    train_dataset, dataset = dataset['train'], dataset['test']


    # 4. DPO 训练配置
    training_args = DPOConfig(
        output_dir=new_model,
        eval_steps=50,
        per_device_eval_batch_size=16,
        bf16=True,
        report_to="wandb",
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
        log_level="debug",
        seed=3047,
    )

    # 5. 初始化 DPOTrainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        # compute_metrics=compute_metrics,
    )

    # 6. 运行评估
    dpo_trainer.evaluate()

    metrics = dpo_trainer.evaluate()

    # 7. 记录评估结果
    print("Evaluation completed. Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}\n")

def load_trained(new_model):
    # 1. 加载基础模型 & Tokenizer
    model, tokenizer = get_model(
        model_name=model_name,
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
        },
    )

    # 3. 使用 `model` 直接加载 PeftModel
    model = PeftModel.from_pretrained(model, new_model)

    return model, tokenizer

if __name__ == "__main__":

    trained_model, tokenizer = load_trained(
        new_model="DPO_NeuralHermes-2.5-Mistral-7B/checkpoint-500"
    )

    evaluate(
        model="teknium/OpenHermes-2.5-Mistral-7B",
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir" :cache_dir,
        },
        dataset="antropic",
        peft_config="DPO_NeuralHermes-2.5-Mistral-7B/checkpoint-500"
    )



    # model, tokenizer = get_model(
    #     model_name="teknium/OpenHermes-2.5-Mistral-7B",
    #     model_config={
    #         "torch_dtype": torch.float16,
    #         "load_in_4bit": True,
    #         "cache_dir" :cache_dir,
    #     },
    # )

    # output = generate_response(
    #     model=model,
    #     tokenizer=tokenizer,
    #     input_text="The biggest mountain in the world is?",
    #     max_length=1000
    # )

    # print(output)

    # intel_dataset = get_intel_imdb_dataset(tokenizer, 'train')[:10]

    # prompts = intel_dataset['prompt']

    # chosen = intel_dataset['chosen']

    # rejected = intel_dataset['rejected']

    # for i in range(10):

    #     prompt = prompts[i]

    #     output = generate_response(
    #         model=model,
    #         tokenizer=tokenizer,
    #         input_text=prompt,
    #         max_length=1000
    #     )

    #     trained_output = generate_response(
    #         model=trained_model,
    #         tokenizer=tokenizer,
    #         input_text=prompt,
    #         max_length=1000
    #     )

    #     print(
    #         f"output: {output}",
    #         f"trained output: {trained_output}",
    #     )




