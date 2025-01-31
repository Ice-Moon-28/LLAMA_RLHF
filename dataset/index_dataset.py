from torch.utils.data import Dataset

class IndexedDataset(Dataset):
    def __init__(self, dataset, indices):
        """
        dataset: 原始数据集
        indices: 选定的索引列表
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]  # 只访问选中的索引
