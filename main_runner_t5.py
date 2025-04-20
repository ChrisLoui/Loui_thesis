import os
import json
import pyaudio
import threading
import queue
import time
import cv2
from vosk import Model, KaldiRecognizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from concurrent.futures import ThreadPoolExecutor

# ---------------------------
# Load the Vosk Speech Recognition Model
# ---------------------------
# For Mac
vosk_model_path = "/Users/wincedelafuente/Desktop/Loui_thesis/model"
# For WINDOWS (if needed)
# vosk_model_path = "C:\\Users\\kcbar\\OneDrive\\Desktop\\LOUI_THESIS_REAL\\Loui_thesis\\model"
print(f"Loading Vosk model from: {vosk_model_path}")
vosk_model = Model(vosk_model_path)
rec = KaldiRecognizer(vosk_model, 16000)

# ---------------------------
# Load the T5 Model for English-to-FSL Translation
# ---------------------------
t5_model_path = "./t5-finetuned"
print(f"Loading T5 model from: {t5_model_path}")
tokenizer = T5Tokenizer.from_pretrained(t5_model_path)
model_t5 = T5ForConditionalGeneration.from_pretrained(t5_model_path)

# Global queue and state variables
animation_queue = queue.Queue()
frame_queue = queue.Queue()  # Queue for frames to be displayed by the main thread
stop_system = threading.Event()  # Flag to stop all threads

# Add this at the start of your script as a global cache
frame_cache = {}


def preload_frames():
    """Preload all frames into memory for faster retrieval"""
    DATA_PATH = 'Step_2/MY_DATA'
    for action in os.listdir(DATA_PATH):
        action_path = os.path.join(DATA_PATH, action)
        if os.path.isdir(action_path):
            frame_cache[action] = collect_frames_for_action(action)
    print("Frames preloaded into cache")


def post_process_fsl(translation):
    # Remove unwanted prefix if present and do any additional text fixes
    if translation.startswith("FSL:"):
        translation = translation.replace("FSL:", "", 1).strip()
    return translation


def translate_english_to_fsl(english_sentence, max_length=50):
    """
    Translates a given English sentence into Filipino Sign Language (FSL) gloss using the fine-tuned T5 model.
    """
    # Use the training prefix for English-to-FSL translation.
    input_text = "translate english to fsl: " + english_sentence
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", max_length=128, truncation=True)
    output_ids = model_t5.generate(
        input_ids, max_length=max_length, num_beams=5, early_stopping=True)
    translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return post_process_fsl(translation)


def collect_frames_for_action(action, DATA_PATH='Step_2/MY_DATA', sequence=None):
    """
    Optimized frame collection with caching
    """
    # Return cached frames if available
    if action in frame_cache:
        return frame_cache[action]

    frames_list = []
    action_folder = os.path.join(DATA_PATH, action)

    if not os.path.exists(action_folder):
        print(f"Action folder '{action_folder}' does not exist.")
        return frames_list

    sequences = [seq for seq in os.listdir(action_folder) if seq.isdigit()]
    if sequence is not None:
        sequences = [str(sequence)]

    # Use list comprehension for faster processing
    frames_list = [(cv2.imread(os.path.join(action_folder, seq, frame)), action)
                   for seq in sequences
                   for frame in sorted(os.listdir(os.path.join(action_folder, seq)))
                   if frame.endswith('.jpg')]

    # Store in cache
    frame_cache[action] = frames_list
    return frames_list


def resize_frame(frame, scale_percent=50):
    """Resize frame to improve performance"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def player_thread():
    """
    Thread that monitors the animation queue, collects frames,
    and adds them to the frame queue for display by the main thread.
    """
    while not stop_system.is_set():
        try:
            # Get the next sentence from the queue (wait up to 0.5 seconds)
            sentence = animation_queue.get(timeout=0.5)

            try:
                # Split into words and process each one
                words = sentence.split()
                for word in words:
                    if stop_system.is_set():
                        break

                    try:
                        # Collect frames for this word
                        frames = collect_frames_for_action(word, sequence=0)

                        if not frames:
                            print(f"No frames found for '{word}'")
                            continue

                        # Add frames to the frame queue for display by the main thread
                        for frame_data in frames:
                            frame_queue.put(frame_data)

                        # Add a marker to indicate end of this word's frames
                        frame_queue.put(None)

                    except Exception as e:
                        print(f"Error processing frames for '{word}': {e}")
            finally:
                # Mark task as done
                animation_queue.task_done()

        except queue.Empty:
            # No items in the queue, just continue checking
            pass


def speech_recognition_thread():
    """
    Thread that handles continuous speech recognition.
    """
    # Setup PyAudio to capture audio from the microphone
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000,
                    input=True, frames_per_buffer=8000)
    stream.start_stream()

    try:
        while not stop_system.is_set():
            data = stream.read(4000, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "")
                if text:
                    print("Transcribed text:", text)
                    # Translate to FSL gloss using the T5 model
                    translation = translate_english_to_fsl(text)
                    print("FSL Gloss Translation:", translation)

                    # Add the translation to the queue
                    animation_queue.put(translation)
            else:
                partial_result = json.loads(rec.PartialResult())
                print(partial_result.get("partial", ""), end="\r")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


def display_frames_thread(fps=30):
    """Optimized frame display thread"""
    current_window = None
    frame_time = 1/fps
    last_time = time.time()

    while not stop_system.is_set():
        try:
            frame_data = frame_queue.get(timeout=0.1)

            if frame_data is None:
                if current_window is not None:
                    cv2.destroyWindow(current_window)
                    current_window = None
                continue

            frame, action = frame_data

            # Resize frame if needed
            frame = resize_frame(frame)

            window_name = f'Action: {action}'
            current_window = window_name

            # Frame timing control
            current_time = time.time()
            elapsed = current_time - last_time
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)

            cv2.imshow(window_name, frame)
            last_time = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_system.set()
                break

        except queue.Empty:
            pass


def process_frames_parallel(words):
    """Process multiple words' frames in parallel"""
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(collect_frames_for_action, word)
                   for word in words]
        return [future.result() for future in futures]


def main():
    print("Starting concurrent FSL translation system...")
    print("Preloading frames...")
    preload_frames()  # Preload frames at startup

    print("Listening, translating, and playing frames... (press Ctrl+C to stop)")

    # Start the player thread (collects frames)
    player = threading.Thread(target=player_thread)
    player.daemon = True
    player.start()

    # Start the speech recognition thread
    speech_thread = threading.Thread(target=speech_recognition_thread)
    speech_thread.daemon = True
    speech_thread.start()

    # Display frames in the main thread to avoid OpenCV GUI issues
    try:
        display_frames_thread()
    except KeyboardInterrupt:
        print("\nStopping transcription and frame playback.")
    finally:
        # Signal all threads to stop
        stop_system.set()

        # Wait for threads to finish their current tasks
        speech_thread.join(timeout=2)
        player.join(timeout=2)

        # Clear any remaining windows
        cv2.destroyAllWindows()

        print("System stopped.")


if __name__ == "__main__":
    main()
