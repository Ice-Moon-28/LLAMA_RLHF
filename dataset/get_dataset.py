import random
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset.index_dataset import IndexedDataset
from dataset.transform import transform_dataset

def get_antropic_dataset(tokenizer, split="test"):
    dataset = load_dataset("Anthropic/hh-rlhf")[split]

    original_columns = dataset.column_names


    # 预处理数据集，拆分出 prompt、chosen 和 rejected
    def preprocess_antropic_example(example):
        if "Assistant:" in example["chosen"] and "Assistant:" in example["rejected"]:
            prompt = example["chosen"].split("Assistant:")[0].strip()
            chosen_response = example["chosen"].split("Assistant:")[1].strip()
            rejected_response = example["rejected"].split("Assistant:")[1].strip()

        else:
            prompt = example["chosen"]  # 以防数据格式不同
            chosen_response = example["chosen"]
            rejected_response = example["rejected"]

        message = {"role": "user", "content": prompt}
        prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)


        return {"prompt": prompt, "chosen": chosen_response + "<|im_end|>\n", "rejected": rejected_response + "<|im_end|>\n"}

    # 处理整个数据集
    processed_dataset = dataset.map(preprocess_antropic_example, remove_columns=original_columns)

    return processed_dataset


def get_intel_imdb_dataset(tokenizer, split="test"):
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

    # Load dataset
    dataset = load_dataset("Intel/orca_dpo_pairs")[split]

    # Save columns
    original_columns = dataset.column_names

    # Format dataset
    dataset = dataset.map(
        chatml_format,
        remove_columns=original_columns
    )

    return dataset

def split_dataset(dataset, train_ratio=0.9, seed=42):
    """
    返回两个新的 Dataset 实例，而不是 Subset
    """
    total_size = len(dataset)
    indices = list(range(total_size))
    
    random.seed(seed)
    random.shuffle(indices)

    train_size = int(total_size * train_ratio)
    train_indices, val_indices = indices[:train_size], indices[train_size:]

    return {
        "train": transform_dataset(IndexedDataset(dataset, train_indices)),
        "test": transform_dataset(IndexedDataset(dataset, val_indices))
    }

if __name__ == "__main__":
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dataset = get_intel_imdb_dataset(tokenizer=tokenizer, split='train')

    dataset = split_dataset(dataset, 0.9, 3047)

    train_dataset, test_dataset = dataset['train'], dataset['test']

    import pdb; pdb.set_trace()

