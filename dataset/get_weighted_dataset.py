import torch
from torch.utils.data import Dataset
import random

class ProbabilisticMultiDataset(Dataset):
    def __init__(self, datasets, sampling_probs, seed=42):
        """
        在 __getitem__ 时根据概率动态选择数据集。

        参数:
        - datasets: List[Dataset]，多个 PyTorch 数据集
        - sampling_probs: List[float]，每个数据集对应的采样概率，必须归一化 (总和为1)
        - seed: 随机种子，确保结果可复现
        """
        assert len(datasets) == len(sampling_probs), "数据集数量和采样概率数量必须一致"
        assert abs(sum(sampling_probs) - 1.0) < 1e-6, "采样概率之和必须等于 1"

        self.datasets = datasets
        self.sampling_probs = sampling_probs
        self.num_samples = sum(len(ds) for ds in datasets)  # 估算数据总量
        random.seed(seed)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """按概率选择数据集，并从中随机抽取样本"""
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.sampling_probs, k=1)[0]
        dataset = self.datasets[dataset_idx]
        sample_idx = random.randint(0, len(dataset) - 1)  # 随机从选中的数据集中取样
        return dataset[sample_idx]

class DummyDataset(Dataset):
    def __init__(self, name, size=100):
        self.name = name
        self.data = [f"{name}_{i}" for i in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


if __name__ == "__main__":
    # 创建多个原始数据集
    dataset1 = DummyDataset("dataset1", size=100)
    dataset2 = DummyDataset("dataset2", size=200)
    dataset3 = DummyDataset("dataset3", size=300)

    # 设定不同数据集的采样概率
    sampling_probs = [0.2, 0.3, 0.5]  # dataset1 20%，dataset2 30%，dataset3 50%

    # 生成概率采样数据集
    sampled_dataset = ProbabilisticMultiDataset([dataset1, dataset2, dataset3], sampling_probs, seed=42)

    # 测试采样
    print(f"采样数据: {[sampled_dataset[i] for i in range(20)]}")  # 打印前20个样本