from transformers import pipeline

def generate_text(model, tokenizer, prompt: str, max_length: int = 300):
    text_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    result = text_generator(f"<s>[INST] {prompt} [/INST]")
    return result[0]['generated_text']