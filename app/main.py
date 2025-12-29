import cv2
import numpy as np
import time
import adas_pilot  # Compiled Rust library
from ultralytics import YOLO

def main():
    # 1. Load YOLOv8 Model
    print("Loading YOLOv8 model...")
    # model = YOLO('models/yolov8n.pt') 
    model = YOLO('yolov8n.pt') 
    
    # 2. Initialize the Rust Tracker (Week 2)
    tracker = adas_pilot.RustTracker()
    
    # 3. Setup Video Capture
    video_path = "assets/videos/project_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # INITIALIZE TIME HERE to fix NameError
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize for consistent processing
        frame = cv2.resize(frame, (1280, 720))
        
        # Calculate Delta Time (dt) for Rust Physics
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time
        
        # Ensure dt isn't zero to avoid division by zero in Rust
        if dt <= 0: dt = 0.033 

        # --- WEEK 1: LANE DETECTION (RUST) ---
        try:
            lines = adas_pilot.detect_lanes_rust(frame)
            for i in range(lines.shape[0]):
                x1, y1, x2, y2 = map(int, lines[i])
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        except Exception:
            pass

        # --- WEEK 2: OBJECT DETECTION (PYTHON/YOLO) ---
        # Classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        results = model(frame, verbose=False, classes=[2, 3, 5, 7])
        
        detections_for_rust = []
        for result in results:
            for box in result.boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                w = x2 - x1
                h = y2 - y1
                detections_for_rust.append((x1, y1, w, h))

        # --- WEEK 2: TRACKING & PHYSICS (RUST) ---
        # Returns: (id, x, y, w, h, distance, speed, collisiontime)
        tracked_objs = tracker.process_frame(detections_for_rust, dt)

        # --- TRAFFIC LIGHT VISUALIZATION ---
        for obj in tracked_objs:
            oid, x, y, w, h, dist, speed, ttc = obj
            
            # Default: GREEN (Safe)
            color = (0, 255, 0)
            status = "SAFE"
            
            # YELLOW: Close (< 20m) or moderately fast approach (< 5s TTC)
            if ttc < 5.0 or dist < 20.0:
                color = (0, 255, 255)
                status = "CAUTION"
            
            # RED: Danger (< 2.5s TTC)
            if ttc < 2.5:
                color = (0, 0, 255)
                status = "BRAKE!"

            # Draw Bounding Box
            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
            # Draw Status Tag
            cv2.putText(frame, f"ID {oid}: {status}", (int(x), int(y)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw Distance and TTC beneath the box
            metrics = f"{dist:.1f}m | {ttc:.1f}s"
            cv2.putText(frame, metrics, (int(x), int(y+h)+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('ADAS Pilot - Week 2: Object Detection & TTC', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()