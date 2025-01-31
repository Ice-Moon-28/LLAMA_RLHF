import torch
import torch.nn.functional as F

def compute_log_probs(logits, input_ids, prompt_len):
    """
    计算 `completion` 部分的 log probability
    """

    import pdb; pdb.set_trace()

    log_probs = F.log_softmax(logits, dim=-1)  # 计算 log-softmax
    input_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # 提取 token 的 log-probability

    prompt_len = prompt_len.item() if isinstance(prompt_len, torch.Tensor) else prompt_len

    
    # 仅计算 completion 部分的概率和
    completion_log_probs = input_log_probs[:, prompt_len:].sum(dim=-1)
    return completion_log_probs