```markdown
# Real-Time English to Filipino Sign Language (FSL) Translation System

Welcome to the world of instant translation magic! Ever wished your spoken words could burst into dynamic Filipino Sign Language right before your eyes? Look no further—this machine learning marvel listens, translates, and even shows off its signing skills in real time. Get ready to experience communication like never before!

## Features

- **Live-Action Speech Recognition:** Powered by Vosk, our system listens to your every word.
- **T5-Powered Translation:** Our fine-tuned T5 model transforms your spoken English into crisp, clear FSL gloss.
- **Visual Sign Language Display:** Watch pre-recorded sign frames light up as your message is translated.
- **Continuous Input & Smooth Playback:** Chat away—our queued translation system keeps up with your pace.
- **FSL Grammar Savvy:** Adheres to FSL rules for that authentic feel.
- **Extra Fun:** It’s like having your own personal sign language interpreter—minus the coffee breaks!

## Project Structure

```
Loui_thesis/
├── models/
│   ├── t5-finetuned/   # Your finely-tuned T5 model (the translation wizard)
│   └── vosk-model/     # The speech recognition engine that never sleeps
├── data/
│   ├── train.txt       # 384 FSL translation pairs—your secret sauce
│   └── MY_DATA/        # A treasure trove of sign language video frames
└── src/
    ├── t5_translator.py        # Where the translation magic happens (training)
    ├── t5_translator_runner.py # Time to show off your model (inference)
    ├── frame_player.py         # The stage for your visual sign language performance
    ├── live_translate.py       # The main event: live translation in action
    └── gpt4_translator.py      # An alternative translator for when you feel fancy
```

## Setup

Ready to dive in? Here’s your backstage pass:

1. **Create & Activate Your Virtual Environment:**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. **Install All the Goodies:**  
   ```bash
   python3 -m pip install -r requirements.txt
   ```
3. **Download Your Models:**  
   - Drop your Vosk model in `models/vosk-model/`
   - Either place your pre-trained T5 model or train one and put it in `models/t5-finetuned/`

## Usage

### Running the Live Translation System

Unleash the magic with:
```bash
python3 live_translate.py
```
This command will:
- Start listening to your microphone like a pro.
- Transcribe your speech into text continuously.
- Queue up translations faster than you can say “Sign Language!”
- Display corresponding sign language frames in real time.
- Keep the show rolling while animations play on.

### Training the Translation Model

Want to make your model even smarter? Train it with:
```bash
python3 t5_translator.py
```
*(Our model is fed 384 delightful FSL translation pairs from `train.txt`—yum!)*

### Running Translations with Your Trained Model

Time to see your creation in action:
```bash
python3 t5_translator_runner.py
```

## FSL Translation Rules

Our system follows these cool FSL grammar conventions:
1. **Skip the Fluff:** Omit articles and auxiliary verbs (a, an, the, is, are).
2. **Time/Location First:** Kick off with time or location markers.
3. **Index Like a Pro:** Use indexing pronouns (ME, YOU, HE, SHE, THEY).
4. **Mark the Tense:** Use PAST or FUTURE to keep it clear.
5. **Questions on Point:** Place question words at the end.
6. **All Caps for Impact:** Everything is loud and clear in ALL-CAPS.

*Example:*  
- **English:** "What is your name?"  
- **FSL:** "NAME YOU WHAT"

## Training Data Format

Your `train.txt` file is the secret recipe. It contains pairs in the format:
```
FSL GLOSS => English sentence
```
*Examples:*  
```
YOU NAME WHAT => what is your name?
ME LIVE MANILA => i live in manila
TIME FUTURE ME GO SCHOOL => i am going to school
```

## Requirements

- **Python 3.8+**
- **PyTorch**
- **Transformers Library**
- **OpenCV**
- **Vosk**
- **PyAudio**
- **Microphone Access** (your gateway to live translation magic)

## Recent Improvements

- [x] Queued translation system for smooth, continuous speech recognition
- [x] Threading implemented for simultaneous translation and display
- [x] 384 FSL translation pairs added to training data
- [x] Dedicated runner for model inference
- [x] Enhanced error handling for model loading
- [x] Alternative GPT-4 translator option added (just for fun)

## Future Improvements

- [ ] Expand the FSL sign dataset—more signs, more fun!
- [ ] Boost translation accuracy to near-perfection
- [ ] Tackle more complex sentences
- [ ] Slash latency in real-time translation
- [ ] Craft a sleek user interface for ultimate control

## Known Issues

- Speech recognition might need a tweak in noisy environments.
- Some signs might be missing in the current dataset.
- Translation may lag slightly on longer sentences—patience is a virtue!

## Contributing

Feel like adding your own magic? Here’s how to join:
1. Fork the repository.
2. Create your feature branch.
3. Commit your awesome changes.
4. Push to the branch.
5. Open a Pull Request and let the fun begin!

## License

*(Insert license information here if applicable.)*

## Author

**Chris Loui A. Canete**

## Acknowledgments

A big shout-out to:
- [Vosk](https://alphacephei.com/vosk/) for their stellar speech recognition.
- [Hugging Face](https://huggingface.co/) for powering our T5 model.
- [MediaPipe](https://mediapipe.dev/) for their innovative pose detection.

---

