from transformers import T5ForConditionalGeneration, T5Tokenizer

# Path to your fine-tuned model directory
model_path = "./t5-finetuned"

# Load the tokenizer and model from the fine-tuned directory
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def translate_english_to_fsl(english_sentence, max_length=50):
    """
    Translates a given English sentence into Filipino Sign Language (FSL) gloss using the fine-tuned model.

    Args:
        english_sentence (str): English sentence to translate.
        max_length (int): Maximum length of the generated output.

    Returns:
        str: Translated FSL gloss.
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

# Example usage
if __name__ == "__main__":
    # Sample English sentence (use one of your training examples or new data)
    english_sentence = "what"
    translation = translate_english_to_fsl(english_sentence)
    print("English Input: ", english_sentence)
    print("FSL Translation: ", translation)
