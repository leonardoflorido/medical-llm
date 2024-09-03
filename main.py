from scripts.fine_tune import fine_tune
from scripts.inference import generate_text

def main():
    model_name = "aboonaji/llama2finetune-v2"
    train_dataset_path = "aboonaji/wiki_medical_terms_llam2_format"
    output_dir = "./results"
    
    # Fine-tune the model
    model, tokenizer = fine_tune(model_name, train_dataset_path, output_dir)
    
    # Example interaction with the model
    user_prompt = "Please tell me about Ascariasis"
    response = generate_text(model, tokenizer, user_prompt)
    
    print(response)

if __name__ == "__main__":
    main()