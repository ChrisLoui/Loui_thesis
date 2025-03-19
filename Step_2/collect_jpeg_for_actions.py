import json
import os
import cv2
import numpy as np


def load_checkpoint(checkpoint_file="../Json_files/checkpoint.json"):
    """
    Load checkpoint information from a JSON file.
    Expected format:
    {
        "action": "hello"
    }
    Returns the action if the file exists; otherwise returns None.
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            return checkpoint.get("action")
    return None


def save_checkpoint(action, checkpoint_file="../Json_files/checkpoint.json"):
    checkpoint = {"action": action}
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)


def collect_jpeg_for_actions():
    # Load checkpoint if it exists
    checkpoint_action = load_checkpoint("../Json_files/checkpoint.json")

    cap = cv2.VideoCapture(0)
    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Wait for 'k' key press before starting
    print("Press 'k' to start collection or 'q' to quit")
    waiting_for_key = True

    while waiting_for_key:
        ret, frame = cap.read()
        if not ret:
            continue

        # Add instructions to the frame
        cv2.putText(frame, "Press 'k' to start collection", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('OpenCV Feed', frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('k'):
            waiting_for_key = False
            print("Starting collection...")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    paused = False
    # If a checkpoint exists, skip actions until we reach that action.
    start_processing = True if checkpoint_action is None else False

    # Iterate over each action
    for action in actions:
        # Skip until we reach the checkpoint action.
        if not start_processing:
            if action != checkpoint_action:
                print(
                    f"Skipping action '{action}', waiting for checkpoint action '{checkpoint_action}'.")
                continue
            else:
                start_processing = True
                print(
                    f"Resuming at checkpoint action '{action}'. Starting fresh collection.")

        # Create the folder with an uppercase name for the action
        action_path = os.path.join(DATA_PATH, action.upper())
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        # Use a single sequence folder named "0" for each action
        sequence = "0"
        sequence_path = os.path.join(action_path, sequence)
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)
        else:
            # Remove all existing JPEG files to start fresh for this action.
            for f in os.listdir(sequence_path):
                if f.endswith('.jpg'):
                    os.remove(os.path.join(sequence_path, f))

        # Start fresh collection
        start_frame = 0

        print(
            f"Collecting data for '{action}' starting at frame {start_frame}/{sequence_length}")

        frame_count = start_frame
        while frame_count < sequence_length:
            ret, frame = cap.read()
            if not ret:
                continue

            # Create a copy for display and add text overlays on that copy.
            display_frame = frame.copy()

            # Text to be centered
            context_text = f'Collecting for Seq {sequence}'
            cv2.putText(display_frame, context_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)

            # Display the action in large, bold text in the center
            # Get the text size (width and height) for proper centering
            (text_width, text_height), baseline = cv2.getTextSize(
                action.upper(), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)

            # Calculate the position to center the action text
            center_x = (frame.shape[1] - text_width) // 2  # Horizontal center
            center_y = (frame.shape[0] + text_height) // 2  # Vertical center

            # Draw a semi-transparent background for better visibility
            # This creates a darker rectangle behind the text
            overlay = display_frame.copy()
            cv2.rectangle(overlay,
                          (center_x - 20, center_y - text_height - 20),
                          (center_x + text_width + 20, center_y + 20),
                          (0, 0, 0), -1)
            # Apply the overlay with transparency
            alpha = 0.6  # Transparency factor
            cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)

            # Now draw the large action text
            cv2.putText(display_frame, action.upper(), (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

            if frame_count == start_frame:
                # Additionally display the "STARTING COLLECTION" text
                cv2.putText(display_frame, 'STARTING COLLECTION', (120, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', display_frame)
                cv2.waitKey(2000)  # Wait for 2 seconds before continuing
            else:
                # Just show the frame with the action text
                cv2.imshow('OpenCV Feed', display_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('p'):
                paused = not paused
                while paused:
                    cv2.putText(display_frame, 'PAUSED', (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', display_frame)
                    if cv2.waitKey(100) & 0xFF == ord('p'):
                        paused = False
            if paused:
                continue

            # Save the raw frame (without text overlays)
            output_file = os.path.join(sequence_path, f"{frame_count}.jpg")
            cv2.imwrite(output_file, frame)
            frame_count += 1

            # Save checkpoint after each frame (saving only the current action)
            save_checkpoint(action)

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Path for exported data, JPEG images
    DATA_PATH = 'MY_DATA'

    # Actions to collect
    actions = np.array(['hello', 'me', 'doctor', 'how', 'you', '...'])

    # Each sequence will contain 30 frames
    sequence_length = 30

    # Print actions to verify
    print(actions)

    # Run the collection process
    collect_jpeg_for_actions()
