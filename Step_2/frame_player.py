import os
import cv2
import argparse
import sys

def play_collected_frames_for_action(action, DATA_PATH='MY_DATA', fps=30, sequence=None):
    action_folder = os.path.join(DATA_PATH, action)
    if not os.path.exists(action_folder):
        print(f"Action folder '{action_folder}' does not exist.")
        return False  # Return False to indicate action doesn't exist or playback was interrupted

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

            cv2.imshow(f'Action: {action}', frame)
            key = cv2.waitKey(int(1000/fps)) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                print("Quit requested. Stopping playback.")
                return False  # Return False to indicate playback was interrupted

        cv2.destroyAllWindows()

    return True  # Return True to indicate successful playback

def play_all_actions(DATA_PATH='MY_DATA', fps=30, sequence=None, start_action=None, end_action=None):
    """Play action folders found in the data path, optionally within a range."""
    if not os.path.exists(DATA_PATH):
        print(f"Data path '{DATA_PATH}' does not exist.")
        return

    # Get all folders in the data path
    action_folders = sorted([folder for folder in os.listdir(DATA_PATH)
                     if os.path.isdir(os.path.join(DATA_PATH, folder))])

    if not action_folders:
        print(f"No action folders found in '{DATA_PATH}'.")
        return

    # Handle start and end actions
    start_idx = 0
    end_idx = len(action_folders) - 1

    if start_action:
        start_action = start_action.upper()
        if start_action in action_folders:
            start_idx = action_folders.index(start_action)
        else:
            print(f"Start action '{start_action}' not found. Starting from the beginning.")

    if end_action:
        end_action = end_action.upper()
        if end_action in action_folders:
            end_idx = action_folders.index(end_action)
        else:
            print(f"End action '{end_action}' not found. Playing until the end.")

    # Ensure valid range
    if start_idx > end_idx:
        print("Start action comes after end action in the folder list. Aborting.")
        return

    actions_to_play = action_folders[start_idx:end_idx+1]

    print(f"Playing {len(actions_to_play)} actions from '{actions_to_play[0]}' to '{actions_to_play[-1]}'...")

    # Play each action
    for action in actions_to_play:
        print(f"Playing action: {action}")
        # If playback was interrupted by 'q', stop playing further actions
        if not play_collected_frames_for_action(action, DATA_PATH, fps, sequence):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play FSL action frames')
    parser.add_argument('--action', type=str, help='Specific action to play', default=None)
    parser.add_argument('--all', action='store_true', help='Play all actions')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second')
    parser.add_argument('--start', type=str, help='Start playing from this action (inclusive)', default=None)
    parser.add_argument('--end', type=str, help='End playing at this action (inclusive)', default=None)

    args = parser.parse_args()

    if args.action:
        # Play a specific action
        play_collected_frames_for_action(args.action.upper(), fps=args.fps)
    elif args.all or args.start or args.end:
        # Play all actions or a range
        play_all_actions(fps=args.fps, start_action=args.start, end_action=args.end)
    else:
        # No valid options provided
        print("No action specified. Use --action NAME to play a specific action or --all to play all actions.")
        print("Examples:")
        print("  python frame_player.py --action HELLO")
        print("  python frame_player.py --all")
        print("  python frame_player.py --start HELLO --end THANK")