# Real-Time English to Filipino Sign Language (FSL) Translation System

A machine learning system that translates spoken English to Filipino Sign Language (FSL) in real-time, using speech recognition, T5-based translation, and visual sign language display.

## Features

- Real-time speech recognition using Vosk
- English to FSL gloss translation using fine-tuned T5 model
- Visual display of sign language using pre-recorded frames
- Support for continuous speech input and translation
- Follows FSL grammar rules and conventions
- Queued translation system for smooth playback

## Project Structure


bash
Loui_thesis/
├── models/
│ ├── t5-finetuned/ # Trained T5 model
│ └── vosk-model/ # Speech recognition model
├── data/
│ ├── train.txt # FSL translation pairs (384 examples)
│ └── MY_DATA/ # Sign language video frames
└── src/
├── t5_translator.py # Translation model training
├── t5_translator_runner.py # Model inference
├── frame_player.py # Visual display
├── live_translate.py # Main application
└── gpt4_translator.py # Alternative GPT-4 translator

## Setup

1. Create and activate virtual environment:
bash
python3 -m venv venv
source venv/bin/activate
2. Install required packages:
bash
python3 -m pip install -r requirements.txt
3. Download required models:
- Place Vosk model in `models/vosk-model/`
- Place or train T5 model in `models/t5-finetuned/`

## Usage

### Running the Live Translation System
bash
python3 live_translate.py
This will:
- Start listening to your microphone
- Transcribe speech to text continuously
- Queue translations for processing
- Display corresponding sign language frames
- Continue listening while playing animations

### Training the Translation Model
bash
python3 t5_translator.py

The model is trained on 384 FSL translation pairs from train.txt.

### Running Translations with Trained Model
bash
python3 t5_translator_runner.py

## FSL Translation Rules

The system follows these FSL grammar conventions:
1. Omits articles and auxiliary verbs (a, an, the, is, are)
2. Places time/location markers at the beginning
3. Uses indexing pronouns (ME, YOU, HE, SHE, THEY)
4. Marks tense with PAST or FUTURE
5. Places question words at the end
6. Uses ALL-CAPS notation

Example:
- English: "What is your name?"
- FSL: "NAME YOU WHAT"

## Training Data Format

The `train.txt` file contains translation pairs:
text
FSL GLOSS => English sentence
Examples:
text
YOU NAME WHAT => what is your name?
ME LIVE MANILA => i live in manila
TIME FUTURE ME GO SCHOOL => i am going to school


## Requirements

- Python 3.8+
- PyTorch
- Transformers library
- OpenCV
- Vosk
- PyAudio
- Microphone access

## Recent Improvements

- [x] Added queued translation system for continuous speech recognition
- [x] Implemented threading for simultaneous translation and display
- [x] Added 384 FSL translation pairs for training
- [x] Created separate runner for model inference
- [x] Improved error handling in model loading
- [x] Added alternative GPT-4 translator option

## Future Improvements

- [ ] Add more FSL signs to the dataset
- [ ] Improve translation accuracy
- [ ] Add support for more complex sentences
- [ ] Reduce latency in real-time translation
- [ ] Add user interface for better control

## Known Issues

- Speech recognition may need adjustment in noisy environments
- Some signs may not be available in the current dataset
- Translation may have slight delays with long sentences

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License


## Author

Chris Loui A. Canete

## Acknowledgments

- [Vosk](https://alphacephei.com/vosk/) for speech recognition
- [Hugging Face](https://huggingface.co/) for T5 model
- [MediaPipe](https://mediapipe.dev/) for pose detection