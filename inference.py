from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import time

def translate_to_fsl(english_text):
    # Get absolute path to model directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "t5-finetuned")
    
    print(f"Looking for model in: {model_path}")
    print(f"Directory exists: {os.path.exists(model_path)}")
    if os.path.exists(model_path):
        print("Files in directory:")
        for file in os.listdir(model_path):
            print(f"- {file}")
    
    # Load the model and tokenizer from the fine-tuned directory
    print("Loading tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    print("Loading model...")
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    # Move model to CPU and set to evaluation mode
    model = model.to('cpu')
    model.eval()
    
    # Prepare the input
    prefix = "translate english to fsl gloss translation: "
    input_text = prefix + english_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
    
    # Measure latency
    start_time = time.time()
    
    # Generate translation with optimized settings for CPU
    with torch.no_grad():  # Disable gradient calculation
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            early_stopping=True,
            do_sample=False,  # Disable sampling for faster generation
            use_cache=True    # Enable KV-caching
        )
    
    end_time = time.time()
    latency = end_time - start_time
    print(f"Translation latency: {latency:.4f} seconds")
    
    # Decode and return the translation
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translation

if __name__ == "__main__":
    # Test the translation
    test_text = "I am going to the store"
    print(f"English: {test_text}")
    print(f"FSL: {translate_to_fsl(test_text)}") 