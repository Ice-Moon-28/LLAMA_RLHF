from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
from tqdm import tqdm

from dataset.get_dataset import get_antropic_dataset
from model.get_model import get_model
from eval.metrics.rouge_score import compute_rouge_bert_metrics_batch
from util.generate_text import generate_response
from torch.utils.data import DataLoader
from settings import device

def evaluate(
        model,
        model_config,
        dataset,
        peft_config=None,    
    ):

    model, tokenizer = get_model(
        model_name=model,
        model_config=model_config
    )

    model.to(device)

    if dataset == 'antropic':
        eval_dataset = get_antropic_dataset(tokenizer=tokenizer)
    
    def tokenize_function(examples):
        prompts = [example["prompt"] for example in examples]

        chosen = [example["chosen"] for example in examples]

        rejected = [example["rejected"] for example in examples]

        tokenized_prompt = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )

        tokenized_prompt = tokenizer(prompts,padding=True,truncation=True,max_length=1024,return_tensors="pt")

        # tokenized_chosen = tokenizer(
        #     examples["chosen"],
        #     padding=True,
        #     truncation=True,
        #     max_length=512,
        #     return_tensors="pt"
        # )

        # tokenized_rejected = tokenizer(
        #     examples["rejected"],
        #     padding=True,
        #     truncation=True,
        #     max_length=512,
        #     return_tensors="pt"
        # )

        return {
            "prompt": {
                "input_ids": tokenized_prompt["input_ids"],
                "attention_mask": tokenized_prompt["attention_mask"],
            },
            "chosen": chosen,
            "rejected": rejected,
        }

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=16, 
        shuffle=False, 
        collate_fn=tokenize_function
    )

    if peft_config:
        model = PeftModel.from_pretrained(model, peft_config)

    for batch in tqdm(eval_dataloader, desc="Generating Responses"):

        responses = generate_response(
            example=batch,
            model=model,
            tokenizer=tokenizer,
        )

        import pdb; pdb.set_trace()

        compute_rouge_bert_metrics_batch(responses)
