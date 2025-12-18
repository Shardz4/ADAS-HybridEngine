import cv2
import numpy as np
import adas_pilot  # Compiled Rust library

def main():
    cap = cv2.VideoCapture("assets/videos/project_video.mp4")
    if not cap.isOpened():
        print("Error: Could not open video file. Ensure 'highway.mp4' is in assets/videos/.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for consistent processing (1280x720)
        frame = cv2.resize(frame, (1280, 720))

        # Call Rust for lane detection
        try:
            lines = adas_pilot.detect_lanes_rust(frame)
            if lines.shape[0] >= 2:  # Expect at least 2 lines (left, right)
                for i in range(min(2, lines.shape[0])):
                    x1, y1, x2, y2 = map(int, lines[i])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
        except Exception as e:
            print(f"Rust call failed: {e}")

        cv2.imshow('ADAS Pilot - Week 1: Lane Detection', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 30 FPS
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()