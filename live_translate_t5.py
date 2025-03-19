import os
import json
import pyaudio
from transformers import T5ForConditionalGeneration, T5Tokenizer
from vosk import Model, KaldiRecognizer
import torch


# ---------------------------
# Load the Vosk Speech Recognition Model
# ---------------------------
# For Mac
vosk_model_path = "/Users/wincedelafuente/Desktop/Loui_thesis/model"
# For WINDOWS
# vosk_model_path = "C:\\Users\\kcbar\\OneDrive\\Desktop\\LOUI_THESIS_REAL\\Loui_thesis\\model"
print(f"Loading Vosk model from: {vosk_model_path}")
vosk_model = Model(vosk_model_path)
rec = KaldiRecognizer(vosk_model, 16000)

# Setup PyAudio to capture audio from the microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8000)
stream.start_stream()

# ---------------------------
# Load the T5 Model for English-to-FSL Translation
# ---------------------------
t5_model_path = "./t5-finetuned"
print(f"Loading T5 model from: {t5_model_path}")
tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
model_t5 = T5ForConditionalGeneration.from_pretrained(t5_model_path)

def post_process_fsl(translation):
    # Remove unwanted prefix if present and do any additional text fixes
    if translation.startswith("FSL:"):
        translation = translation.replace("FSL:", "", 1).strip()
    translation = translation.replace("YOUR", "YOU")
    return translation

def translate_english_to_fsl(english_sentence, max_length=50):
    """
    Translates a given English sentence into Filipino Sign Language (FSL) gloss
    using the fine-tuned T5 model with greedy decoding.
    """
    input_text = "translate english to fsl: " + english_sentence
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        # Using greedy decoding (num_beams=1) to speed up translation
        output_ids = model_t5.generate(input_ids, max_length=max_length, num_beams=1)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return post_process_fsl(translation)

def main():
    print("Listening and translating... (press Ctrl+C to stop)")
    try:
        while True:
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print("Transcribed text:", text)
                    # Join detected words to form the English sentence
                    english_sentence = " ".join(text.split())
                    # Translate to FSL gloss using the T5 model
                    translation = translate_english_to_fsl(english_sentence)
                    print("FSL Gloss Translation:", translation)
            else:
                partial_result = json.loads(rec.PartialResult())
                print(partial_result.get("partial", ""), end="\r")
    except KeyboardInterrupt:
        print("\nStopping transcription and translation.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
