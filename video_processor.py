from Step_1.utils import *


def process_video_file_all_frames(video_path, holistic, show_feedback=False,
                                  no_hand_threshold=20, max_frames=60):
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    started = False
    no_hand_counter = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    wait_time = int(1000 / fps)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image, results = mediapipe_detection(frame, holistic)

        if show_feedback:
            draw_styled_landmarks(image, results)
            cv2.imshow('Video Processing', image)
            key = cv2.waitKey(wait_time) & 0xFF
            if key == ord('q'):
                break

        if (results.left_hand_landmarks is not None) or (results.right_hand_landmarks is not None):
            if not started:
                started = True
                print("Hand detected. Starting frame collection...")

            no_hand_counter = 0
            frames_list.append(frame.copy())

            if len(frames_list) >= max_frames:
                print(f"Collected {max_frames} frames. Stopping this video.")
                break
        else:
            if started:
                no_hand_counter += 1
                print(
                    f"No hand detected for {no_hand_counter} consecutive frames.")
                if no_hand_counter >= no_hand_threshold:
                    print(
                        f"Reached {no_hand_threshold} consecutive frames with no hand detected. Stopping.")
                    break
            continue

    cap.release()
    if show_feedback:
        cv2.destroyWindow('Video Processing')

    return frames_list


def process_videos_for_actions(DATA_PATH='MY_DATA', video_root='clips',
                               actions_to_process=None, no_hand_threshold=20,
                               max_frames=60, show_feedback=True):
    all_actions = [d for d in os.listdir(video_root)
                   if os.path.isdir(os.path.join(video_root, d))]
    all_actions.sort()
    print("Found actions:", all_actions)

    if isinstance(actions_to_process, str):
        actions_to_process = [actions_to_process]

    with mp_holistic.Holistic(min_detection_confidence=0.3,
                              min_tracking_confidence=0.3) as holistic:
        for action in all_actions:
            if actions_to_process is not None and action not in actions_to_process:
                print(
                    f"Skipping '{action}' since it's not in {actions_to_process}")
                continue

            action_save_path = os.path.join(DATA_PATH, action)
            os.makedirs(action_save_path, exist_ok=True)

            action_video_path = os.path.join(video_root, action)
            if not os.path.exists(action_video_path):
                print(
                    f"Video folder for action '{action}' does not exist. Skipping.")
                continue

            video_files = [f for f in os.listdir(action_video_path)
                           if f.lower().endswith(('.mp4', '.mov'))]
            video_files.sort()

            for idx, video_file in enumerate(video_files):
                video_path = os.path.join(action_video_path, video_file)
                print(
                    f"\nProcessing {video_file} (index {idx}) for action '{action}'...")

                frames_sequence = process_video_file_all_frames(
                    video_path, holistic, show_feedback=show_feedback,
                    no_hand_threshold=no_hand_threshold, max_frames=max_frames
                )

                if not frames_sequence:
                    print(f"No valid frames found in {video_file}. Skipping.")
                    continue

                sequence_save_path = os.path.join(action_save_path, str(idx))
                os.makedirs(sequence_save_path, exist_ok=True)

                for frame_num, frame in enumerate(frames_sequence):
                    image_path = os.path.join(
                        sequence_save_path, f"{frame_num}.jpg")
                    cv2.imwrite(image_path, frame)

                print(
                    f"Saved {len(frames_sequence)} frames for action '{action}' from {video_file}.")

    print("\nVideo processing complete.")


if __name__ == "__main__":
    process_videos_for_actions()
