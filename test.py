from transformers import AutoTokenizer

from dataset.get_dataset import get_antropic_dataset

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

        
