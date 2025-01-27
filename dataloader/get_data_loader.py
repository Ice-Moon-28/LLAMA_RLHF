import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


def create_dataloader(dataset, tokenizer, data_collate_fn, batch_size=8, shuffle=True, max_prompt_length=1024, max_length=1536):
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
    
    # def preprocess_function(batch):
    #     # 将文本分为 prompt 和 completion，假设数据集具有相应字段
    #     prompt_texts = batch.get("prompt", "")
    #     completion_texts = batch.get("completion", "")
        
    #     # 分词并截断
    #     prompt_encodings = tokenizer(
    #         prompt_texts,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=max_prompt_length,
    #         return_tensors="pt"
    #     )
        
    #     completion_encodings = tokenizer(
    #         completion_texts,
    #         padding="max_length",
    #         truncation=True,
    #         max_length=max_length - max_prompt_length,  # 剩余长度留给 completion
    #         return_tensors="pt"
    #     )
        
    #     # 合并编码
    #     input_ids = torch.cat([prompt_encodings["input_ids"], completion_encodings["input_ids"]], dim=1)
    #     attention_mask = torch.cat([prompt_encodings["attention_mask"], completion_encodings["attention_mask"]], dim=1)
        
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask,
    #         # 如果有标签，可以添加标签字段
    #         "labels": input_ids.clone(),  # 假设标签与输入相同，用于自回归任务
    #     }
    
    # # 如果数据未被预处理，则对其进行分词预处理
    # dataset = dataset.map(preprocess_function, batched=True)

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=data_collate_fn,
    )
    return dataloader



if __name__ == "__main__":
    dataset = load_dataset("Intel/orca_dpo_pairs")['train']
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    # Save columns
    original_columns = dataset.column_names

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def chatml_format(example):
        # Format system
        if len(example['system']) > 0:
            message = {"role": "system", "content": example['system']}
            system = tokenizer.apply_chat_template([message], tokenize=False)
        else:
            system = ""

        # Format instruction
        message = {"role": "user", "content": example['question']}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)

        # Format chosen answer
        chosen = example['chosen'] + "<|im_end|>\n"

        # Format rejected answer
        rejected = example['rejected'] + "<|im_end|>\n"

        return {
            "prompt": system + prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    # Format dataset
    dataset = dataset.map(
        chatml_format,
        remove_columns=original_columns,

    )

    dataloader = create_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
    )