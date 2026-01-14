import cv2
import numpy as np
import time
import os # Import OS to check file paths
import adas_pilot
from ultralytics import YOLO

# CONFIG
IS_TWO_WAY_ROAD = False  

def main():
    print("[DEBUG] Starting App...")
    
    # 1. Load Model
    print("[DEBUG] Loading YOLOv8 model...")
    try:
        model = YOLO('yolov8n.pt')
        print("[DEBUG] Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        return

    # 2. Init Rust
    print("[DEBUG] Initializing Rust modules...")
    tracker = adas_pilot.RustTracker()
    manager = adas_pilot.RustLaneManager(smoothing=0.6, is_two_way=IS_TWO_WAY_ROAD)
    print("[DEBUG] Rust modules initialized.")

    # 3. Check Video Path
    video_path = "assets/videos/project_video.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] Video file NOT FOUND at: {os.path.abspath(video_path)}")
        print("Please check if the file is inside the 'assets/videos' folder.")
        return

    print(f"[DEBUG] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("[ERROR] cv2.VideoCapture failed to open the file (Codec error?).")
        return

    print("[DEBUG] Video opened. Starting loop...")
    
    prev_time = time.time()
    active_left, active_right = None, None
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("[DEBUG] Loop ended: No more frames (or failed to read).")
            break
            
        frame_count += 1
        if frame_count % 30 == 0: print(f"[DEBUG] Processing frame {frame_count}...")

        frame = cv2.resize(frame, (1280, 720))
        h, w, _ = frame.shape
        
        cur_time = time.time()
        dt = cur_time - prev_time
        prev_time = cur_time
        if dt <= 0: dt = 0.033

        # --- STEP 1: DETECT & SMOOTH LANES ---
        try:
            raw_lines_np = adas_pilot.detect_lanes_rust(frame)
            raw_lines_list = [tuple(x) for x in raw_lines_np]
            l_tup, r_tup = manager.update_lanes(raw_lines_list, float(w))
            
            if l_tup != (0.,0.,0.,0.): active_left = l_tup
            if r_tup != (0.,0.,0.,0.): active_right = r_tup
        except Exception as e:
            print(f"[WARN] Lane detection error: {e}")

        # --- STEP 2: YOLO ---
        results = model(frame, verbose=False, classes=[2, 3, 5, 7])
        raw_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                raw_detections.append((x1, y1, x2-x1, y2-y1))

        # --- STEP 3: FILTER ---
        valid_detections = manager.filter_objects(raw_detections)

        # --- STEP 4: TRACK ---
        tracked_objs = tracker.process_frame(valid_detections, dt)

        # --- DRAWING ---
        if active_left:
            color = (0, 255, 255) if IS_TWO_WAY_ROAD else (255, 0, 0)
            cv2.line(frame, (int(active_left[0]), int(active_left[1])), (int(active_left[2]), int(active_left[3])), color, 3)
        if active_right:
            cv2.line(frame, (int(active_right[0]), int(active_right[1])), (int(active_right[2]), int(active_right[3])), (255, 0, 0), 3)

        for obj in tracked_objs:
            oid, x, y, w, h, dist, speed, ttc = obj
            color = (0, 255, 0)
            if ttc < 5.0 or dist < 20.0: color = (0, 255, 255)
            if ttc < 2.5: color = (0, 0, 255)
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            cv2.putText(frame, f"{dist:.1f}m", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('ADAS Pilot - Debug Mode', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    print("[DEBUG] App finished.")

if __name__ == "__main__":
    main()