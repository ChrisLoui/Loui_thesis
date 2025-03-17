import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import string

# Add GPU check at the start
if not torch.cuda.is_available():
    raise RuntimeError("This script requires a GPU with CUDA support. Please check your PyTorch installation.")

print(f"Using GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

def normalize_fsl_gloss(text: str) -> str:
    """
    Normalize FSL gloss according to selected rules:
      1. Omit articles, prepositions, auxiliary verbs (e.g., A, AN, THE, IS, ARE, DO, DOES, etc.).
      6. Return only ALL-CAPS words separated by single spaces, with no punctuation.
    
    Note: Rules 2-5 (about time/location, indexing pronouns, tense marking, question word positioning) 
          require more context-aware processing and are not fully implemented here.
    """
    # Convert to uppercase
    text = text.upper()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove stop words (simple list for articles, prepositions, auxiliary verbs)
    stop_words = {"A", "AN", "THE", "IS", "ARE", "DO", "DOES", "DID", "HAS", "HAVE", "AM"}
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

class TranslationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Process each line; skip empty lines.
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "=>" in line:
                parts = line.split("=>")
                # Format in file: FSL gloss => English sentence.
                # But we use English as input and FSL gloss as target.
                fsl_text = parts[0].strip()      # FSL gloss (target)
                english_text = parts[1].strip()  # English sentence (input)
                self.inputs.append(english_text)
                # Normalize FSL gloss to follow the given rules
                normalized_fsl = normalize_fsl_gloss(fsl_text)
                self.targets.append(normalized_fsl)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Use a prefix to instruct translation.
        prefix = "translate english to fsl:"
        input_text = prefix + self.inputs[idx]
        target_text = self.targets[idx]

        inputs = self.tokenizer.encode_plus(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        targets = self.tokenizer.encode_plus(
            target_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # Squeeze to remove extra batch dimension.
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        labels = targets['input_ids'].squeeze()
        # Replace padding token IDs with -100 so that they are ignored by the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    device = torch.device('cuda')  # We already ensured CUDA is available.
    print(f"Using device: {device}")
    print(f"GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    model_name = "t5-base"
    print(f"Loading model {model_name}...")
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Move model to GPU
    model = model.to(device)
    print(f"Model loaded and moved to {device}")
    print(f"GPU Memory allocated after model load: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print("Loading dataset...")
    dataset = TranslationDataset("train.txt", tokenizer)
    print(f"Dataset loaded with {len(dataset)} examples")

    training_args = TrainingArguments(
        output_dir="./t5-finetuned",
        num_train_epochs=20,
        per_device_train_batch_size=2,  # Adjust based on your GPU memory
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=10,
        save_total_limit=2,
        evaluation_strategy="no",
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4,  # Helps with memory usage
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting training...")
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.save_model("./t5-finetuned")
    tokenizer.save_pretrained("./t5-finetuned")
    print("Fine-tuning complete. Model saved to './t5-finetuned'.")
    print(f"Final GPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
