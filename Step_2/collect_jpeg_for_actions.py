from Step_1.utils import *
import json

def load_checkpoint(checkpoint_file="checkpoint.json"):
    """
    Load checkpoint information from a JSON file.
    Expected format:
    {
        "action": "hello",
        "sequence": 0
    }
    Returns a tuple (action, sequence) if the file exists; otherwise returns (None, 0).
    """
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint = json.load(f)
            return checkpoint.get("action"), checkpoint.get("sequence")
    return None, 0

def save_checkpoint(action, sequence, checkpoint_file="checkpoint.json"):
    checkpoint = {"action": action, "sequence": sequence}
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint, f)

def collect_jpeg_for_actions():
    # Load checkpoint if it exists
    checkpoint_action, checkpoint_sequence = load_checkpoint("checkpoint.json")

    cap = cv2.VideoCapture(0)
    # Set desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    paused = False
    with mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        for action in actions:
            action_path = os.path.join(DATA_PATH, action)
            if not os.path.exists(action_path):
                os.makedirs(action_path)
            # Determine starting sequence: if this action matches the checkpoint, resume; else, start from 0
            if checkpoint_action == action:
                start_sequence = checkpoint_sequence
            else:
                # Check existing non-empty sequence directories
                def get_start_sequence(action_path):
                    existing = []
                    for seq in os.listdir(action_path):
                        if seq.isdigit():
                            seq_path = os.path.join(action_path, seq)
                            npy_files = [f for f in os.listdir(seq_path) if f.endswith('.npy')]
                            if npy_files:
                                existing.append(int(seq))
                    return max(existing) + 1 if existing else 0
                start_sequence = get_start_sequence(action_path)

            print(f"Collecting data for '{action}' starting at sequence {start_sequence}/{no_sequences}")

            for sequence in range(start_sequence, no_sequences):
                sequence_path = os.path.join(action_path, str(sequence))
                if not os.path.exists(sequence_path):
                    os.makedirs(sequence_path)

                frame_count = 0
                while frame_count < sequence_length:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    image, results = mediapipe_detection(frame, holistic)
                    draw_styled_landmarks(image, results)

                    # Provide visual feedback
                    if frame_count == 0:
                        cv2.putText(image, 'STARTING COLLECTION', (120,200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4, cv2.LINE_AA)
                        cv2.putText(image, f'Collecting for {action} Seq {sequence}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        cv2.waitKey(1500)
                    else:
                        cv2.putText(image, f'Collecting for {action} Seq {sequence}', (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)

                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('p'):
                        paused = not paused
                        while paused:
                            cv2.putText(image, 'PAUSED', (200,200),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
                            cv2.imshow('OpenCV Feed', image)
                            if cv2.waitKey(100) & 0xFF == ord('p'):
                                paused = False
                    if paused:
                        continue

                    # Check if detection is complete; if not, skip frame
                    if results.pose_landmarks is None:
                        cv2.putText(image, 'Incomplete detection. Adjust position.', (50,50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        continue

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(sequence_path, f"{frame_count}.npy")
                    np.save(npy_path, keypoints)
                    frame_count += 1

                    # Save checkpoint after each frame
                    save_checkpoint(action, sequence)

                    if key == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path for exported data, numpy arrays
    DATA_PATH = 'MY_DATA'

    # Actions to collect
    actions = np.array(['hello', 'one', 'two', 'three', 'four'])

    # Thirty videos worth of data
    no_sequences = 2

    # Videos are going to be 30 frames in length
    sequence_length = 30

    # Print actions to verify
    print(actions)

    # Run collection
    collect_jpeg_for_actions()