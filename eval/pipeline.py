
from peft import LoraConfig, PeftModel
import torch
from transformers import AutoModelForCausalLM

from data.get_data_loader import get_data_loader


def pipeline():
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    new_model = "DPO_NeuralHermes-2.5-Mistral-7B"
    cache_dir = ""

    # Model to fine-tune
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        cache_dir=cache_dir,
    )

    model.config.use_cache = False


    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    model = PeftModel(model, peft_config)

    dataloader = get_data_loader(batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for batch in dataloader:
        # 将 batch 数据转移到设备
        chosen = {
            "input_ids": batch["chosen"]["input_ids"].to(device),
            "attention_mask": batch["chosen"]["attention_mask"].to(device)
        }

        rejected = {
            "input_ids": batch["rejected"]["input_ids"].to(device),
            "attention_mask": batch["rejected"]["attention_mask"].to(device)
        }

        # 使用 PEFT 包装的模型进行前向传播
        chosen_output = model(**chosen)
        rejected_output = model(**rejected)

        # 调试或其他操作
        import pdb; pdb.set_trace()
        print("Chosen Output:", chosen_output)
        print("Rejected Output:", rejected_output)