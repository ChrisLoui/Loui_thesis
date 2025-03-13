import pyaudio
import json
from vosk import Model, KaldiRecognizer
import os
from pathlib import Path

# Load the model with absolute path
model_path = "/Users/wincedelafuente/Desktop/Loui_thesis/model"
print(f"Loading model from: {model_path}")
model = Model(model_path)
rec = KaldiRecognizer(model, 16000)

# Setup PyAudio to capture audio from the microphone
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
stream.start_stream()

print("Listening... (press Ctrl+C to stop)")

try:
    while True:
        data = stream.read(4000, exception_on_overflow=False)
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            print(result.get("text", ""))
        else:
            partial_result = json.loads(rec.PartialResult())
            print(partial_result.get("partial", ""), end="\r")
except KeyboardInterrupt:
    print("\nStopping transcription.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()