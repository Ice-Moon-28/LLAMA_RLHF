from datasets import Dataset
import torch

def transform_dataset(train_dataset):
    # 1. 如果是 PyTorch Dataset，需要手动转换为 List[Dict]
    if isinstance(train_dataset, torch.utils.data.Dataset):
        train_dataset = list(train_dataset)  # ✅ 转换为列表

    # 2. 直接转换为 Hugging Face Dataset
    train_dataset = Dataset.from_list(train_dataset)  # ✅ 直接转换

    return train_dataset