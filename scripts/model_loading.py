import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_name: str):
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        quantization_config=config
    )
    
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    
    return model