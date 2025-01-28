import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from imdb import get_imdb_collect_fn, imdb_pre_process

model_name = "teknium/OpenHermes-2.5-Mistral-7B"
new_model = "DPO_NeuralHermes-2.5-Mistral-7B"
cache_dir = "/root/autodl-tmp"
# Fine-tune model with DPO
def create_dataloader(dataset, collect_fn, batch_size=8, shuffle=True):
    """
    创建一个 DataLoader，用于处理给定的数据集。

    参数:
        dataset (Dataset): 输入的数据集，需支持 __getitem__ 和 __len__。
        tokenizer (AutoTokenizer): 用于对文本进行分词的 tokenizer。
        data_collate_fn (function): 数据收集器函数，用于将数据批量化处理。
        batch_size (int): 每个 batch 的大小（默认值为 8）。
        shuffle (bool): 是否对数据进行随机打乱（默认值为 True）。
        max_prompt_length (int): prompt 的最大长度（默认值为 1024）。
        max_length (int): 输入样本的总最大长度（默认值为 1536）。
    
    返回:
        DataLoader: 用于批量处理数据的 PyTorch DataLoader 对象。
    """
    
    
    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collect_fn,
    )
    return dataloader



if __name__ == "__main__":
    dataset = load_dataset("Intel/orca_dpo_pairs")['train']
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    # Save columns
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dataset = imdb_pre_process(dataset, tokenizer)

    dataloader = create_dataloader(
        dataset=dataset,
        collect_fn=get_imdb_collect_fn(tokenizer=tokenizer)
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        cache_dir=cache_dir,
    )
    model.config.use_cache = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有 GPU 可用

    for batch in dataloader:
       
        chosen = {
            "input_ids": batch["chosen"]["input_ids"].to(device),
            "attention_mask": batch["chosen"]["attention_mask"].to(device)
        }

        rejected = {
            "input_ids": batch["rejected"]["input_ids"].to(device),
            "attention_mask": batch["rejected"]["attention_mask"].to(device)
        }

        output = model(**chosen)

        import pdb; pdb.set_trace()
        print(batch)

