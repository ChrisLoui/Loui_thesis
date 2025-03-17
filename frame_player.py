import os
import cv2


def play_collected_frames_for_action(action, DATA_PATH='Step_2/MY_DATA', fps=10, sequence=None):
    action_folder = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_folder):
        print(f"Action folder '{action_folder}' does not exist.")
        return

    sequences = [seq for seq in os.listdir(action_folder) if seq.isdigit()]
    if sequence is not None:
        sequences = [str(sequence)]

    for seq in sequences:
        seq_folder = os.path.join(action_folder, seq)
        frames = [f for f in os.listdir(seq_folder) if f.endswith('.jpg')]
        frames.sort(key=lambda x: int(x.split('.')[0]))

        for frame_file in frames:
            frame_path = os.path.join(seq_folder, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            cv2.imshow(f'Action: {action}, Sequence: {seq}', frame)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    play_collected_frames_for_action("doctor".upper(), fps=30, sequence=None)
