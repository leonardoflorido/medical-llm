from scripts.model_loading import load_model
from scripts.tokenizer_loading import load_tokenizer
from scripts.training import set_training_arguments, create_trainer

def fine_tune(model_name, train_dataset_path, output_dir, batch_size=4, max_steps=100):
    model = load_model(model_name)
    tokenizer = load_tokenizer(model_name)
    
    training_args = set_training_arguments(output_dir, batch_size, max_steps)
    
    trainer = create_trainer(model, tokenizer, train_dataset_path, training_args)
    trainer.train()

    return model, tokenizer