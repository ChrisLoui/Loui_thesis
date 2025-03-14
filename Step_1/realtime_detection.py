from Step_1.utils import *


def run_realtime_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('OpenCV Feed', 1280, 720)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame_failures = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                frame_failures += 1
                print(f"Frame not captured, failure count: {frame_failures}")
                if frame_failures > 10:
                    print("Too many failures, stopping capture.")
                    break
                continue

            frame_failures = 0
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            cv2.imshow('OpenCV Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_realtime_detection()
