import torch

def get_imdb_collect_fn(tokenizer, max_prompt_length=1024, max_length=1536):
    def preprocess_function(batch):
        prompt_texts = [item.get("prompt", "") for item in batch]  # 提取 prompt 字段
        chosen_texts = [item.get("chosen", "") for item in batch]  # 提取 chosen 字段
        rejected_texts = [item.get("rejected", "") for item in batch]  # 提取 rejected 字段

        # 分词并截断
        # 处理 prompt
        prompt_encodings = tokenizer(
            prompt_texts,
            padding="max_length",
            truncation=True,
            max_length=max_prompt_length,
            return_tensors="pt"  # 返回 PyTorch 张量
        )
        
        # 处理 chosen
        chosen_texts_encodings = tokenizer(
            chosen_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length - max_prompt_length,  # 剩余长度留给 completion
            return_tensors="pt"
        )

        # 处理 rejected
        rejected_texts_encodings = tokenizer(
            rejected_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length - max_prompt_length,  # 剩余长度留给 completion
            return_tensors="pt"
        )
        
        # 合并编码
        return {
            "chosen": {
                "input_ids": torch.cat([prompt_encodings["input_ids"], chosen_texts_encodings["input_ids"]], dim=1),
                "attention_mask": torch.cat([prompt_encodings["attention_mask"], chosen_texts_encodings["attention_mask"]], dim=1),
            },
            "rejected": {
                "input_ids": torch.cat([prompt_encodings["input_ids"], rejected_texts_encodings["input_ids"]], dim=1),
                "attention_mask": torch.cat([prompt_encodings["attention_mask"], rejected_texts_encodings["attention_mask"]], dim=1),
            }
        }

    return preprocess_function


def imdb_pre_process(dataset, tokenizer):
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

    original_columns = dataset.column_names
    dataset = dataset.map(
        chatml_format,
        remove_columns=original_columns,
    )

    return dataset