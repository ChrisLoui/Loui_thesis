import csv
import time
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
from tqdm import tqdm

# Download necessary NLTK packages if not already available
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('punkt')

# Path to your fine-tuned model directory
model_path = "./t5-finetuned"

# Load the tokenizer and model from the fine-tuned directory
print("Loading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded and moved to {device}")

def translate_english_to_fsl(english_sentence, max_length=50):
    """
    Translates a given English sentence into FSL gloss using the fine-tuned model.
    Returns the translation and the time taken.
    """
    # Use the prefix that was used during training for English-to-FSL translation.
    input_text = "translate english to fsl: " + english_sentence
    
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
    
    # Measure translation time
    start_time = time.time()
    
    # Generate the translation
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, 
            max_length=max_length, 
            num_beams=5, 
            early_stopping=True
        )
    
    # Calculate latency
    latency = time.time() - start_time
    
    # Decode the output tokens to a string
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return translation, latency

def calculate_bleu(reference, hypothesis):
    """
    Calculate BLEU score for a single sentence.
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    # Apply smoothing to handle zero matches
    smoothie = SmoothingFunction().method1
    
    # Calculate BLEU-1 (unigrams), BLEU-2 (bigrams), BLEU-3 (trigrams), BLEU-4 (4-grams)
    bleu1 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu2 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu3 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu([reference_tokens], hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return {
        "bleu1": bleu1,
        "bleu2": bleu2, 
        "bleu3": bleu3,
        "bleu4": bleu4
    }

def calculate_meteor(reference, hypothesis):
    """
    Calculate METEOR score for a single sentence.
    """
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    
    # Calculate METEOR score
    return meteor_score([reference_tokens], hypothesis_tokens)

def load_validation_data(file_path):
    """
    Loads validation data from a CSV file.
    The CSV should have two columns: 'english' and 'fsl'
    """
    return pd.read_csv(file_path)

def run_validation(validation_data):
    """
    Runs the validation by comparing model output with expected FSL glosses.
    """
    results = []
    total_latency = 0
    
    print(f"Running validation on {len(validation_data)} examples...")
    
    for _, row in tqdm(validation_data.iterrows(), total=len(validation_data)):
        english_sentence = row["english"]
        expected_fsl = row["fsl"]
        
        # Translate and measure latency
        predicted_fsl, latency = translate_english_to_fsl(english_sentence)
        total_latency += latency
        
        # Calculate metrics
        bleu_scores = calculate_bleu(expected_fsl, predicted_fsl)
        meteor = calculate_meteor(expected_fsl, predicted_fsl)
        
        # Check for exact match
        exact_match = predicted_fsl.strip() == expected_fsl.strip()
        
        # Store results
        results.append({
            "english": english_sentence,
            "expected_fsl": expected_fsl,
            "predicted_fsl": predicted_fsl,
            "latency": latency,
            "bleu1": bleu_scores["bleu1"],
            "bleu2": bleu_scores["bleu2"],
            "bleu3": bleu_scores["bleu3"],
            "bleu4": bleu_scores["bleu4"],
            "meteor": meteor,
            "exact_match": exact_match
        })
    
    # Convert results to DataFrame for easy analysis
    results_df = pd.DataFrame(results)
    
    # Calculate summary statistics
    avg_latency = total_latency / len(validation_data)
    exact_match_accuracy = results_df["exact_match"].mean()
    avg_bleu1 = results_df["bleu1"].mean()
    avg_bleu2 = results_df["bleu2"].mean()
    avg_bleu3 = results_df["bleu3"].mean()
    avg_bleu4 = results_df["bleu4"].mean()
    avg_meteor = results_df["meteor"].mean()
    
    # Display detailed results for each example
    print("\n=== Individual Translation Examples ===")
    for i, result in enumerate(results[:5]):  # Show first 5 examples
        print(f"\nExample {i+1}:")
        print(f"English Input:    {result['english']}")
        print(f"Expected FSL:     {result['expected_fsl']}")
        print(f"Predicted FSL:    {result['predicted_fsl']}")
        print(f"BLEU-1:           {result['bleu1']:.4f}")
        print(f"BLEU-4:           {result['bleu4']:.4f}")
        print(f"METEOR:           {result['meteor']:.4f}")
        print(f"Latency:          {result['latency']:.4f} seconds")
        print("-" * 40)
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total examples:         {len(validation_data)}")
    print(f"Exact match accuracy:   {exact_match_accuracy:.2%}")
    print(f"Average BLEU-1 score:   {avg_bleu1:.4f}")
    print(f"Average BLEU-2 score:   {avg_bleu2:.4f}")
    print(f"Average BLEU-3 score:   {avg_bleu3:.4f}")
    print(f"Average BLEU-4 score:   {avg_bleu4:.4f}")
    print(f"Average METEOR score:   {avg_meteor:.4f}")
    print(f"Average latency:        {avg_latency:.4f} seconds")
    
    # Save detailed results to CSV
    results_df.to_csv("validation_results.csv", index=False)
    print("\nDetailed results saved to 'validation_results.csv'")
    
    # Return the results DataFrame for further analysis if needed
    return results_df

if __name__ == "__main__":
    validation_file = "validation.csv"  # This file must exist with the correct format
    validation_data = load_validation_data(validation_file)
    results = run_validation(validation_data)