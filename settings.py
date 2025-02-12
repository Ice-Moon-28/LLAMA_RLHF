import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检查是否有 GPU 可用