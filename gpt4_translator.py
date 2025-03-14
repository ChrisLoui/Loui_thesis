import os
import openai
from dotenv import load_dotenv


def generate_sentence_from_words_gpt4(detected_words, openai_api_key):
    """
    Translates detected sign language words into a proper sentence using GPT-4.

    Args:
        detected_words (list): List of detected sign language words
        openai_api_key (str): OpenAI API key for GPT-4 access

    Returns:
        str: Translated FSL gloss sentence
    """
    openai.api_key = openai_api_key
    gloss = " ".join(detected_words)

    prompt = (
        "You are an expert translator specializing in translating English grammar into "
        "Filipino Sign Language (FSL) gloss notation. Given an English sentence, provide "
        "exactly one accurate Filipino Sign Language gloss translation, structured according "
        "to the grammatical rules of FSL. Return only a sequence of English words separated "
        "by a single space, with no additional punctuation or symbols. "
        f"English Input:\n{gloss}\n\n"
        "Filipino Sign Language Gloss Translation:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert translator."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=20,
        n=1,
        stop=["\n"]
    )

    translation = response["choices"][0]["message"]["content"].strip()
    return translation


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()

    # Get API key from environment variable
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # Example usage
    detected_words = ["hello", "two"]
    translation = generate_sentence_from_words_gpt4(
        detected_words, openai_api_key)
    print(f"Detected words: {detected_words}")
    print(f"Translated gloss: {translation}")
