import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Path to your fine-tuned model directory
model_path = "./t5-finetuned"

# Load the tokenizer and model from the fine-tuned directory
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def translate_english_to_fsl(english_sentence, max_length=50):
    """
    Translates a given English sentence into FSL gloss using the fine-tuned model.
    """
    # Use the prefix that was used during training for English-to-FSL translation.
    input_text = "translate English to FSL: " + english_sentence
    # Encode the input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    # Generate the translation
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    # Decode the output tokens to a string
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return translation

def load_validation_data(file_path):
    """
    Loads validation data from a CSV file.
    The CSV should have two columns: 'english' and 'fsl'
    """
    data = []
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({"english": row["english"], "fsl": row["fsl"]})
    return data

def run_validation(validation_data):
    """
    Runs the validation by comparing model output with expected FSL glosses.
    """
    num_correct = 0
    total = len(validation_data)
    
    for pair in validation_data:
        english_sentence = pair["english"]
        expected_fsl = pair["fsl"]
        predicted_fsl = translate_english_to_fsl(english_sentence)
        
        print("English Input:    ", english_sentence)
        print("Expected FSL:     ", expected_fsl)
        print("Predicted FSL:    ", predicted_fsl)
        print("-" * 40)
        
        # Compare after stripping whitespace. For a more lenient evaluation,
        # consider fuzzy matching or BLEU score.
        if predicted_fsl.strip() == expected_fsl:
            num_correct += 1

    accuracy = num_correct / total
    print("Exact Match Accuracy: {:.2%}".format(accuracy))

if __name__ == "__main__":
    validation_file = "validation.csv"  # This file must exist with the correct format.
    validation_data = load_validation_data(validation_file)
    run_validation(validation_data)
