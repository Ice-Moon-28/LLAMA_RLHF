from transformers import AutoTokenizer, AutoModelForCausalLM

def get_model(
    model_name = "teknium/OpenHermes-2.5-Mistral-7B",
    model_config=None,

):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_config,
    )
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer
