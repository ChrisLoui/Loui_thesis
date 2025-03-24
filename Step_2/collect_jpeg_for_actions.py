import json
import os
import cv2
import numpy as np
import time


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


def load_word_list(word_list_file="../Json_files/words_list.json"):
    """
    Load the list of words from a JSON file.
    Expected format:
    {
        "doctor_patient_words": ["a", "b", "c", ...]
    }
    Returns the list of words if the file exists; otherwise returns a default list.
    """
    if os.path.exists(word_list_file):
        with open(word_list_file, "r") as f:
            word_data = json.load(f)
            return word_data.get("doctor_patient_words", [])
    print(
        f"Warning: Word list file '{word_list_file}' not found. Using default words.")
    return ["hello", "me", "doctor", "how", "you"]


def draw_countdown_timer(frame, seconds_left):
    """
    Draw a visual countdown timer on the frame.
    """
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Draw a large circle in the center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 6

    # Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.circle(overlay, (center_x, center_y), radius, (0, 0, 200), -1)

    # Add the number in the center
    font_scale = 2.0
    thickness = 10
    (text_width, text_height), baseline = cv2.getTextSize(
        str(seconds_left), cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

    # Center the text in the circle
    text_x = center_x - text_width // 2
    text_y = center_y + text_height // 2

    cv2.putText(overlay, str(seconds_left), (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Blend the overlay with the original frame
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Add "GET READY" text above the timer
    message = "GET READY"
    font_scale_msg = 1.5
    thickness_msg = 3
    (msg_width, msg_height), _ = cv2.getTextSize(
        message, cv2.FONT_HERSHEY_SIMPLEX, font_scale_msg, thickness_msg)

    # Position the message above the circle
    msg_x = center_x - msg_width // 2
    msg_y = center_y - radius - 20

    cv2.putText(frame, message, (msg_x, msg_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_msg, (0, 255, 0), thickness_msg, cv2.LINE_AA)

    return frame


def collect_jpeg_for_actions():
    # Load checkpoint if it exists
    checkpoint_action = load_checkpoint("../Json_files/checkpoint.json")

    # Load the list of words from JSON
    actions = load_word_list("../Json_files/words_list.json")
    print(f"Loaded {len(actions)} words from JSON file.")
    print(f"First 5 words: {actions[:5]}")

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

        # Display countdown timer before starting to collect frames
        countdown_duration = 2  # 2 seconds countdown
        countdown_start = time.time()

        while time.time() - countdown_start < countdown_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            # Calculate remaining seconds
            seconds_left = max(1, int(countdown_duration -
                               (time.time() - countdown_start) + 1))

            # Create a copy for display
            display_frame = frame.copy()

            # Add action information
            cv2.putText(display_frame, f"Preparing for: {action.upper()}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            # Draw countdown timer
            display_frame = draw_countdown_timer(display_frame, seconds_left)

            cv2.imshow('OpenCV Feed', display_frame)

            # Check for quit key
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Start actual frame collection
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
            cv2.addWeighted(overlay, alpha, display_frame,
                            1 - alpha, 0, display_frame)

            # Now draw the large action text
            cv2.putText(display_frame, action.upper(), (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3, cv2.LINE_AA)

            # Add frame counter
            cv2.putText(display_frame, f"Frame: {frame_count+1}/{sequence_length}", (frame.shape[1]-250, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            if frame_count == start_frame:
                # Additionally display the "STARTING COLLECTION" text
                cv2.putText(display_frame, 'STARTING COLLECTION', (120, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', display_frame)
                cv2.waitKey(1000)  # Wait for 1 second before continuing
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

    # Each sequence will contain 30 frames
    sequence_length = 50

    # Run the collection process
    collect_jpeg_for_actions()
