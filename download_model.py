import requests
import zipfile
import os
import sys
from pathlib import Path

# Get the absolute path to the model directory
model_dir = Path(__file__).parent / "model"

def download_model():
    # Create model directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)

    # Download the model
    print("Downloading Vosk English model...")
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    zip_path = model_dir / "model.zip"

    try:
        print(f"Downloading to {model_dir}...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes

        # Download the file
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        downloaded = 0

        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Print progress
                    done = int(50 * downloaded / total_size)
                    sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded}/{total_size} bytes")
                    sys.stdout.flush()

        print("\n\nExtracting model files...")
        # Extract the model
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

        # Remove the zip file
        zip_path.unlink()

        # Move files from the extracted directory to model_dir
        extracted_dir = next(model_dir.glob("vosk-model*"))
        for item in extracted_dir.iterdir():
            target = model_dir / item.name
            if target.exists():
                if target.is_file():
                    target.unlink()
                else:
                    import shutil
                    shutil.rmtree(target)
            item.rename(target)
        extracted_dir.rmdir()

        print("\nModel downloaded and extracted successfully!")
        print(f"Model location: {model_dir}")

    except Exception as e:
        print(f"Error downloading model: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download_model()