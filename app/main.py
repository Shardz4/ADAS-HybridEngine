
import time
import os
import cv2
import numpy as np

try:
    import adas_pilot
    _HAS_RUST = True
except Exception:
    # allow running the UI without the Rust extension (use dummy lines)
    adas_pilot = None
    _HAS_RUST = False


def main():
    video_path = os.path.join(os.path.dirname(__file__), "..", "assets", "videos", "project_video.mp4")
    video_path = os.path.normpath(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"error: Could not open the video file: {video_path}\nEnsure that it exists.")
        return

    target_size = (640, 360)      # smaller size -> faster processing
    frame_skip = 2                # process every 2nd frame
    frame_idx = 0

    last_ts = time.time()
    proc_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize for speed
        frame = cv2.resize(frame, target_size)
        frame_idx += 1

        # Skip some frames to increase playback speed
        if frame_idx % frame_skip != 0:
            cv2.imshow('ADAS Pilot - week1: Lane Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        t0 = time.time()

        # ensure C-contiguous array before passing into Rust
        if not frame.flags['C_CONTIGUOUS']:
            frame = np.ascontiguousarray(frame)

        # Debug: print frame shape once per second
        if int(time.time() - last_ts) == 0:
            pass

        lines = None
        if _HAS_RUST:
            try:
                lines = adas_pilot.detect_lanes_rust(frame)
            except Exception as e:
                print(f"Call failed: {e}")
                lines = None
        # If Rust extension not available or failed, run a Python OpenCV detector
        if lines is None:
            def detect_lanes_py(img):
                # img: BGR uint8
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blur, 50, 150)

                # trapezoidal ROI
                mask = np.zeros_like(edges)
                pts = np.array([[
                    (int(0.1 * w), h),
                    (int(0.4 * w), int(0.6 * h)),
                    (int(0.6 * w), int(0.6 * h)),
                    (int(0.9 * w), h)
                ]], dtype=np.int32)
                cv2.fillPoly(mask, pts, 255)
                masked = cv2.bitwise_and(edges, mask)

                # Hough lines
                lines_p = cv2.HoughLinesP(masked, rho=1, theta=np.pi/180, threshold=50, minLineLength=40, maxLineGap=150)
                if lines_p is None:
                    return np.zeros((0, 4), dtype=np.int32)

                left_lines = []
                right_lines = []
                for l in lines_p:
                    x1, y1, x2, y2 = l[0]
                    if x2 == x1:
                        continue
                    slope = (y2 - y1) / (x2 - x1)
                    if abs(slope) < 0.3:
                        continue
                    if slope < 0:
                        left_lines.append((x1, y1, x2, y2))
                    else:
                        right_lines.append((x1, y1, x2, y2))

                def avg_line(lines_group):
                    if not lines_group:
                        return None
                    x1s = [l[0] for l in lines_group]
                    y1s = [l[1] for l in lines_group]
                    x2s = [l[2] for l in lines_group]
                    y2s = [l[3] for l in lines_group]
                    return (
                        int(sum(x1s) / len(x1s)),
                        int(sum(y1s) / len(y1s)),
                        int(sum(x2s) / len(x2s)),
                        int(sum(y2s) / len(y2s)),
                    )

                avg_left = avg_line(left_lines)
                avg_right = avg_line(right_lines)
                out = []
                if avg_left is not None:
                    out.append(avg_left)
                if avg_right is not None:
                    out.append(avg_right)
                return np.array(out, dtype=np.int32)

            lines = detect_lanes_py(frame)

        # If the rust function returned a numpy array-like, draw the lines
        if getattr(lines, "shape", None) and lines.shape[0] >= 1:
            for i in range(min(2, lines.shape[0])):
                x1, y1, x2, y2 = map(int, lines[i])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 6)

        t1 = time.time()
        proc_frames += 1
        if t1 - last_ts >= 1.0:
            print(f"Processed FPS: {proc_frames/(t1-last_ts):.1f}, last_frame_ms: {(t1-t0)*1000:.0f}ms")
            last_ts = t1
            proc_frames = 0

        cv2.imshow('ADAS Pilot - week1: Lane Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
