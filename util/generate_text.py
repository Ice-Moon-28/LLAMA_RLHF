import torch
from settings import device

def generate_response(example, model, tokenizer, modelSetting={}):
    """
    Given a prompt from the dataset, generate model responses for comparison.
    """
    prompt = example["prompt"]  # The human-preferred response prompt
    inputs = {
        "input_ids": prompt["input_ids"].to(device),
        "attention_mask": prompt["attention_mask"].to(device),
    }
    
    import pdb; pdb.set_trace()

    # Generate response
    with torch.no_grad():
        output = model.generate(**inputs, max_length=512, temperature=0.7, top_p=0.9, **modelSetting)
    
    generated_texts = tokenizer.batch_decode(output, skip_special_tokens=True)
    
    return {
        "prompt": prompt,
        "human_preferred": example["chosen"],
        "model_response": generated_texts,
    }