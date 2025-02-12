import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig
from trl import PPOConfig, PPOTrainer
from torch.utils.data import DataLoader
import wandb

from dataset.get_dataset import get_antropic_dataset, get_intel_imdb_dataset, split_dataset
from dataset.get_weighted_dataset import ProbabilisticMultiDataset
from model.get_model import get_model

# 认证 Token
hf_token = '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'
wb_token = '1c1fa66d79864363e5f33bb705a768da6cf094e5'

# 登录 wandb
wandb.login(key=wb_token)

# 设置模型参数
model_name = "teknium/OpenHermes-2.5-Mistral-7B"
new_model = "PPO_NeuralHermes-2.5-Mistral-7B_WITH_TWO_DATASET"
cache_dir = "/root/autodl-tmp"

def train():
    # 1️⃣ 加载策略模型（Actor）和值模型（Critic）
    actor, tokenizer = get_model(
        model_name=model_name,
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
        },
    )
    
    critic, _ = get_model(
        model_name=model_name,
        model_config={
            "torch_dtype": torch.float16,
            "load_in_4bit": True,
            "cache_dir": cache_dir,
        },
    )

    # 2️⃣ 获取数据集
    intel_dataset = get_intel_imdb_dataset(tokenizer, 'train')
    antropic = get_antropic_dataset(tokenizer, 'train')

    dataset = ProbabilisticMultiDataset(
        datasets=[intel_dataset, antropic],
        sampling_probs=[0.7, 0.3]
    )

    dataset = split_dataset(dataset, 1, 3047)
    train_dataset, eval_dataset = dataset['train'], dataset['test']

    # 3️⃣ LoRA 配置
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj']
    )

    # 4️⃣ PPO 训练参数配置
    training_args = PPOConfig(
        batch_size=4,
        mini_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        total_steps=2000,
        warmup_steps=200,
        output_dir=new_model,
        optim="paged_adamw_32bit",
        bf16=True,
        log_with="wandb",
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        seed=3047,
    )

    # 5️⃣ 创建优化器
    actor_optimizer = optim.AdamW(actor.parameters(), lr=training_args.learning_rate)
    critic_optimizer = optim.AdamW(critic.parameters(), lr=training_args.learning_rate)

    # 6️⃣ 创建 PPO 训练器
    ppo_trainer = PPOTrainer(
        actor_model=actor,
        critic_model=critic,
        config=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    # 7️⃣ 训练循环
    for step in range(training_args.total_steps):
        batch = train_dataset.select(range(training_args.batch_size))
        inputs = tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True).to("cuda")

        # 1️⃣ 计算策略输出（Actor）
        with torch.no_grad():
            logits = actor(**inputs).logits
            actions = torch.argmax(logits, dim=-1)

        # 2️⃣ 计算值函数（Critic）
        values = critic(**inputs).logits.mean(dim=-1)

        # 3️⃣ 计算奖励（示例）
        rewards = compute_rewards(batch["text"])

        # 4️⃣ 计算优势（Advantage）
        advantages = rewards - values.detach()

        # 5️⃣ 更新 Actor（策略模型）
        policy_loss = ppo_loss(actions, logits, advantages)
        actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_optimizer.step()

        # 6️⃣ 更新 Critic（值模型）
        value_loss = nn.MSELoss()(values, rewards)
        critic_optimizer.zero_grad()
        value_loss.backward()
        critic_optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}: Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")

    print("训练完成！")

# 计算 PPO 损失
def ppo_loss(actions, logits, advantages):
    """计算 PPO 策略损失"""
    action_log_probs = torch.log_softmax(logits, dim=-1)
    selected_log_probs = action_log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)
    ratio = torch.exp(selected_log_probs - selected_log_probs.detach())
    clipped_ratio = torch.clamp(ratio, 1 - training_args.cliprange, 1 + training_args.cliprange)
    loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    return loss

# 计算奖励函数
def compute_rewards(texts):
    """示例奖励函数（可以用 LLM 评分、相似度计算等替换）"""
    return torch.tensor([len(text) % 5 + 1 for text in texts], dtype=torch.float16).cuda()  # 伪奖励函数

if __name__ == "__main__":
    train()