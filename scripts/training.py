from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig

def set_training_arguments(output_dir: str, batch_size: int, max_steps: int):
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        max_steps=max_steps
    )

def create_trainer(model, tokenizer, train_dataset_path, training_args):
    train_dataset = load_dataset(path=train_dataset_path, split="train")
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=16,
        lora_dropout=0.1
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        dataset_text_field="text"
    )
    
    return trainer