import cv2
import numpy as np
import time
import adas_pilot
from ultralytics import YOLO

# CONFIG
IS_TWO_WAY_ROAD = False  

def main():
    print("Loading YOLOv8...")
    model = YOLO('yolov8n.pt') 
    
    # 1. Tracker (Physics)
    tracker = adas_pilot.RustTracker()
    
    # 2. Manager (Lanes + Zones)
    # Handles smoothing AND "safe zone" logic in one place
    manager = adas_pilot.RustLaneManager(smoothing=0.6, is_two_way=IS_TWO_WAY_ROAD)
    
    cap = cv2.VideoCapture("assets/videos/highway.mp4")
    prev_time = time.time()

    # Placeholders for drawing
    active_left, active_right = None, None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
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
            
            # One call update lines in Rust.
            # It stores the smooth lines internally for the next step.
            l_tup, r_tup = manager.update_lanes(raw_lines_list, float(w))
            
            if l_tup != (0.,0.,0.,0.): active_left = l_tup
            if r_tup != (0.,0.,0.,0.): active_right = r_tup
        except: pass

        # --- STEP 2: RAW YOLO ---
        results = model(frame, verbose=False, classes=[2, 3, 5, 7])
        raw_detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                raw_detections.append((x1, y1, x2-x1, y2-y1))

        # --- STEP 3: FILTER (Uses Internal Lines) ---
        # No need to pass lines manually! The manager remembers them.
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

        cv2.imshow('ADAS Pilot - Unified Manager', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()