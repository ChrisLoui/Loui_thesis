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

    paused = False
    # Iterate over each action
    for action in actions:
        # Create the folder with an uppercase name for the action
        action_path = os.path.join(DATA_PATH, action.upper())
        if not os.path.exists(action_path):
            os.makedirs(action_path)

        # Use a single sequence folder named "0" for each action
        sequence = "0"
        sequence_path = os.path.join(action_path, sequence)
        if not os.path.exists(sequence_path):
            os.makedirs(sequence_path)

        # Resume frame count if this action matches the checkpoint; otherwise start at 0
        if checkpoint_action == action:
            existing_files = [f for f in os.listdir(
                sequence_path) if f.endswith('.jpg')]
            start_frame = len(existing_files)
        else:
            start_frame = 0

        print(
            f"Collecting data for '{action}' starting at frame {start_frame}/{sequence_length}")

        frame_count = start_frame
        while frame_count < sequence_length:
            ret, frame = cap.read()
            if not ret:
                continue

            # Provide visual feedback
            if frame_count == start_frame:
                cv2.putText(frame, 'STARTING COLLECTION', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, f'Collecting for {action} Seq {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)
                cv2.waitKey(1500)
            else:
                cv2.putText(frame, f'Collecting for {action} Seq {sequence}', (15, 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('p'):
                paused = not paused
                while paused:
                    cv2.putText(frame, 'PAUSED', (200, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', frame)
                    if cv2.waitKey(100) & 0xFF == ord('p'):
                        paused = False
            if paused:
                continue

            # Save the current frame as a JPEG image
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
    actions = np.array(['hello', 'me', 'doctor', 'one', 'two', 'three'])

    # Each sequence will contain 30 frames
    sequence_length = 30

    # Print actions to verify
    print(actions)

    # Run the collection process
    collect_jpeg_for_actions()
