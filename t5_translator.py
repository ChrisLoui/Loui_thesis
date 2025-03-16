import os
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.inputs = []
        self.targets = []
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        # Process each line; skip header if present.
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if "=>" in line:
                parts = line.split("=>")
                # Assuming your training file is formatted as:
                # FSL => English
                fsl_text = parts[0].strip()    # FSL gloss
                english_text = parts[1].strip()  # English sentence
                self.inputs.append(fsl_text)
                self.targets.append(english_text)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Use a prefix that specifies the correct translation direction.
        prefix = "translate English to FSL: "
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
        # Replace padding token id's with -100 so that they are ignored by the loss.
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


def main():
    # You can experiment with larger variants if you have more resources.
    model_name = "t5-"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Load your custom training dataset (ensure your file is formatted as English => FSL).
    dataset = TranslationDataset("train.txt", tokenizer)

    # Define training arguments.
    training_args = TrainingArguments(
        output_dir="./t5-finetuned",
        num_train_epochs=20,
        per_device_train_batch_size=2,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=10,
        save_total_limit=2,
        # You can add evaluation later if you have a validation set.
        evaluation_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    # Start fine-tuning.
    trainer.train()

    # Save the fine-tuned model.
    trainer.save_model("./t5-finetuned")
    tokenizer.save_pretrained("./t5-finetuned")
    print("Fine-tuning complete. Model saved to './t5-finetuned'.")


if __name__ == "__main__":
    main()
