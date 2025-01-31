from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

def evaluate(
        model,
        dataset,
        tokenizer,
        peft_config=None,    
    ):
    # 运行 DPOTrainer 的 evaluate 方法
    training_args = DPOConfig(
        per_device_eval_batch_size=16,
        bf16=True,
        beta=0.1,
        max_prompt_length=1024,
        max_length=1536,
        log_level="debug",
        seed=3047,
    )

    dpo_trainer = DPOTrainer(
        model,
        args=training_args,
        eval_dataset=dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    eval_results = dpo_trainer.evaluate()

    # 打印评估结果
    print("Evaluation Results:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")

