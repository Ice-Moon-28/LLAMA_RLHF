import random
from datasets import load_dataset
from transformers import AutoTokenizer

from dataset.index_dataset import IndexedDataset
from dataset.transform import transform_dataset

def split_text(chosen, reject):
    # 找到从头开始相同的部分
    min_length = min(len(chosen), len(reject))
    split_index = 0
    for i in range(min_length):
        if chosen[i] == reject[i]:
            split_index = i + 1  # 记录分叉点
        else:
            break

    prompt = chosen[:split_index]  # 相同的部分
    chosen_answer = chosen[split_index:]  # chosen 剩余的部分
    reject_answer = reject[split_index:]  # reject 剩余的部分

    return prompt, [chosen_answer, reject_answer]

def split_text_into_turns(text):
    """
    解析多轮对话，将 'User: ... Assistant: ...' 转换成 [{role: "user", content: "..."}, ...]
    """
    messages = []
    lines = text.split("\n")
    current_role = None
    current_content = []

    for line in lines:
        if line == "":
            continue
        if line.startswith("Human:") or line.startswith("User:"):
            if current_content:
                messages.append({"role": current_role, "content": " ".join(current_content)})
                current_content = []
            current_role = "user"
            current_content.append(line.replace("Human:", "").replace("User:", "").strip())
        elif line.startswith("Assistant:"):
            if current_content:
                messages.append({"role": current_role, "content": " ".join(current_content)})
                current_content = []
            current_role = "assistant"
            current_content.append(line.replace("Assistant:", "").strip())
        else:
            current_content.append(line.strip())

    if current_content:
        messages.append({"role": current_role, "content": " ".join(current_content)})

    return messages

def extract_prompt_and_responses(chosen_turns, rejected_turns):
    """
    解析多轮对话，找到相同的对话作为 prompt，分别输出不同的 chosen 和 rejected 作为 AI 的回答。
    
    :param chosen_turns: 解析后的 `chosen` 对话列表（[{role: "user", content: "..."}]）
    :param rejected_turns: 解析后的 `rejected` 对话列表（同上）
    :return: prompt（对话历史）+ chosen_response + rejected_response
    """
    # 找到公共的对话部分（prompt）
    prompt = []
    min_length = min(len(chosen_turns), len(rejected_turns))
    
    for i in range(min_length):
        if chosen_turns[i] == rejected_turns[i]:  # 两者相同，则是公共部分
            prompt.append(chosen_turns[i])
        else:
            break  # 遇到不同部分，停止
    
    # 提取不同的部分作为 AI 的回答
    chosen_response = chosen_turns[len(prompt):]  # chosen 的 AI 回答
    rejected_response = rejected_turns[len(prompt):]  # rejected 的 AI 回答


    return prompt, chosen_response, rejected_response

def get_antropic_dataset(tokenizer, split="test"):
    dataset = load_dataset("Anthropic/hh-rlhf")[split]

    original_columns = dataset.column_names


    def preprocess_anthropic_example(example):
        try:
            # 处理多轮对话
            if "Assistant:" in example["chosen"] and "Assistant:" in example["rejected"]:
                messages_chosen = split_text_into_turns(example["chosen"])
                messages_rejected = split_text_into_turns(example["rejected"])
            else:
                # 兼容单轮对话格式
                messages_chosen = [{"role": "user", "content": example["chosen"]}]
                messages_rejected = [{"role": "user", "content": example["chosen"]}]  # ✅ 修正：应使用 example["rejected"]
                messages_chosen.append({"role": "assistant", "content": example["chosen"]})
                messages_rejected.append({"role": "assistant", "content": example["rejected"]})

            # 提取公共部分（prompt）和不同的 AI 回复
            prompt, chosen_response, rejected_response = extract_prompt_and_responses(messages_chosen, messages_rejected)


            if len(chosen_response) == 0 or len(rejected_response) == 0:
                return {
                    "prompt": '',
                    "chosen": '',
                    "rejected": '',       
                } 

            # 转换为 ChatML 格式
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            chosen_response = tokenizer.apply_chat_template(chosen_response, tokenize=False, add_generation_prompt=False, add_role_prefix=False)
            rejected_response = tokenizer.apply_chat_template(rejected_response, tokenize=False, add_generation_prompt=False, add_role_prefix=False)

            # 移除 <|im_start|>assistant
            if isinstance(chosen_response, str) and chosen_response.startswith("<|im_start|>assistant"):
                chosen_response = chosen_response.replace("<|im_start|>assistant\n", "", 1)  # 仅替换开头

            if isinstance(rejected_response, str) and rejected_response.startswith("<|im_start|>assistant"):
                rejected_response = rejected_response.replace("<|im_start|>assistant\n", "", 1)  # 仅替换开头

            

            return {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,       
            }
        
        except Exception as e:
            print(f"Error in preprocess_anthropic_example: {e}")
            import pdb; pdb.set_trace()
            

    processed_dataset = dataset.map(preprocess_anthropic_example, remove_columns=original_columns).filter(
        lambda x: x["prompt"] != '' and x["chosen"] != '' and x["rejected"] != ''
    )

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

def get_google_natural_dataset():

    dataset = load_dataset("google-research-datasets/natural_questions", "default")

    return dataset


# def get_openai_webgpt_comparisons_dataset(tokenizer, split="test"):
#     dataset = load_dataset("openai/webgpt_comparisons")

#     original_columns = dataset.column_names

#     # 预处理数据集，拆分出 prompt、chosen 和 rejected
#     def preprocess_antropic_example(example):
#         import pdb; pdb.set_trace()
#         if "Assistant:" in example["chosen"] and "Assistant:" in example["rejected"]:
#             prompt = example["chosen"].split("Assistant:")[0].strip()
#             chosen_response = example["chosen"].split("Assistant:")[1].strip()
#             rejected_response = example["rejected"].split("Assistant:")[1].strip()

#         else:
#             prompt = example["chosen"]  # 以防数据格式不同
#             chosen_response = example["chosen"]
#             rejected_response = example["rejected"]

#         message = {"role": "user", "content": prompt}
#         prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=True)


#         return {"prompt": prompt, "chosen": chosen_response + "<|im_end|>\n", "rejected": rejected_response + "<|im_end|>\n"}

#     # 处理整个数据集
#     processed_dataset = dataset.map(preprocess_antropic_example, remove_columns=original_columns)

#     return processed_dataset

if __name__ == "__main__":
    model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    # dataset = get_intel_imdb_dataset(tokenizer=tokenizer, split='train')

    # dataset = split_dataset(dataset, 0.9, 3047)

    # train_dataset, test_dataset = dataset['train'], dataset['test']

    # import pdb; pdb.set_trace()

    
    # get_openai_webgpt_comparisons_dataset(tokenizer=tokenizer)

    dataset = get_antropic_dataset(tokenizer=tokenizer, split="train")

    for item in dataset:
        import pdb; pdb.set_trace()

        


