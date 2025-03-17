import os
import json
import pyaudio
from gpt4_translator import generate_sentence_from_words_gpt4
import openai
from dotenv import load_dotenv
from vosk import Model, KaldiRecognizer

# Load environment variables and API key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Load Vosk model from the absolute path
model_path = "/Users/wincedelafuente/Desktop/Loui_thesis/model"
print(f"Loading model from: {model_path}")
model = Model(model_path)
rec = KaldiRecognizer(model, 16000)

# Setup PyAudio to capture audio from the microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=8000)
stream.start_stream()

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
                    # Split the text into words for translation
                    detected_words = text.split()
                    # Translate to FSL gloss
                    translation = generate_sentence_from_words_gpt4(detected_words, openai_api_key)
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
