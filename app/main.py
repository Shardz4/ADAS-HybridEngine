import cv2
import numpy as np
import time
import os 
import adas_pilot
import onnxruntime as ort
from ultralytics import YOLO
import threading
import queue

# ==========================================
# CONFIGURATION
# ==========================================
IS_TWO_WAY_ROAD = False  
AI_SKIP_FRAMES = 3  

SIGN_CLASSES = {
    52: "STOP",
    # Add more later...
}

# ==========================================
# NEW: MULTI-THREADED CAMERA PIPELINE
# ==========================================
class ThreadedCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        # Create a small waiting room (queue) for 5 frames
        self.q = queue.Queue(maxsize=5) 
        self.stopped = False
        
        # Start the background thread!
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()
        return self

    def update(self):
        # This loop runs constantly in the background
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                
                # We move the resizing to the background thread to free up the Main CPU!
                frame = cv2.resize(frame, (1280, 720))
                self.q.put(frame)
            else:
                # If the waiting room is full, rest for a millisecond
                time.sleep(0.001) 

    def read(self):
        # The main thread just grabs the next ready frame instantly
        return self.q.get()

    def more(self):
        return self.q.qsize() > 0 or not self.stopped

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# ==========================================
# MAIN DASHBOARD
# ==========================================
def main():
    print("--- Starting ADAS Master Suite (Multi-Threaded) ---")
    
    print("[DEBUG] Loading YOLOv8 model for vehicles...")
    try:
        model = YOLO('yolov8n.pt')
        print("[DEBUG] Vehicle Model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO: {e}")
        return

    print("[DEBUG] Initializing Rust modules...")
    try:
        tracker = adas_pilot.RustTracker()
        manager = adas_pilot.RustLaneManager(smoothing=0.6, is_two_way=IS_TWO_WAY_ROAD)
        brain = adas_pilot.AdasBrain("../models/traffic_signs.onnx") 
        print(f"[DEBUG] ONNX Execution Providers: {ort.get_available_providers()}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Rust modules: {e}")
        return

    video_path = "../assets/videos/project_video.mp4" 
    if not os.path.exists(video_path):
        video_path = "test_video.mp4"

    print(f"[DEBUG] Opening multi-threaded stream: {video_path}")
    
    # Start our new threaded camera!
    stream = ThreadedCamera(video_path).start()
    # Give the background thread a second to fill the waiting room
    time.sleep(1.0) 

    if not stream.more():
        print("[ERROR] Failed to open the video file.")
        return

    print("[DEBUG] Streaming live... Press 'q' to quit.")
    
    prev_time = time.time()
    frame_count = 0
    active_left, active_right = None, None
    light_status = "NONE"
    last_sign_detections = []
    last_raw_yolo_detections = []

    # Loop while the background thread is still finding frames
    while stream.more():
        frame = stream.read()
        frame_count += 1
        h, w, _ = frame.shape
        
        cur_time = time.time()
        dt = cur_time - prev_time
        prev_time = cur_time
        if dt <= 0: dt = 0.033

        # ==========================================
        # MEDIUM SUBSYSTEMS (Run Every 2nd Frame)
        # ==========================================
        if frame_count % 2 == 0:
            light_status = adas_pilot.check_traffic_lights(frame)
            try:
                raw_lines_np = adas_pilot.detect_lanes_rust(frame)
                raw_lines_list = [tuple(x) for x in raw_lines_np]
                l_tup, r_tup = manager.update_lanes(raw_lines_list, float(w))
                if l_tup != (0.,0.,0.,0.): active_left = l_tup
                if r_tup != (0.,0.,0.,0.): active_right = r_tup
            except Exception:
                pass

        # ==========================================
        # HEAVY AI SUBSYSTEMS (Run Every N Frames)
        # ==========================================
        if frame_count % AI_SKIP_FRAMES == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                last_sign_detections = brain.process_frame(rgb_frame.tobytes(), w, h, 0.40)
            except Exception:
                pass

            results = model(frame, device='0', half=True, verbose=False, classes=[2, 3, 5, 7])
            last_raw_yolo_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    last_raw_yolo_detections.append((x1, y1, x2-x1, y2-y1))

        # ==========================================
        # TRACKER (Run Every Frame)
        # ==========================================
        filtered_data = manager.filter_objects(last_raw_yolo_detections)
        bboxes_only = []
        ego_map = {} 
        for (bbox, is_ego) in filtered_data:
            bboxes_only.append(bbox)
            ego_map[int(bbox[0])] = is_ego

        tracked_objs = tracker.process_frame(bboxes_only, dt)

        # ==========================================
        # VISUALIZATION HUD
        # ==========================================
        if active_left:
            c = (0,255,0) if IS_TWO_WAY_ROAD else (255,0,0)
            cv2.line(frame, (int(active_left[0]), int(active_left[1])), (int(active_left[2]), int(active_left[3])), c, 3)
        if active_right:
            cv2.line(frame, (int(active_right[0]), int(active_right[1])), (int(active_right[2]), int(active_right[3])), (255,0,0), 3)
        
        light_color = (255, 255, 255) 
        if light_status == "RED": light_color = (0,0,255)
        elif light_status == "YELLOW": light_color = (0,255,255)
        elif light_status == "GREEN": light_color = (0,255,0)

        if light_status != "NONE":
            cv2.rectangle(frame, (20,20) , (250,80), (0,0,0), -1)
            cv2.putText(frame, f"LIGHT: {light_status}", (30,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_color, 3)

        for obj in tracked_objs:
            oid, x, y, bw, bh, dist, speed, ttc = obj
            is_in_ego_lane = ego_map.get(int(x), False)

            color = (0,255,0)
            if dist < 25.0 or (ttc < 5.0 and ttc > 0): color = (0 , 255, 255)
            if is_in_ego_lane and (ttc < 2.5 and ttc > 0): color = (0,0,255)
            
            cv2.rectangle(frame, (int(x), int(y)), (int(x+bw), int(y+bh)), color, 2)
            label_text = f"{dist:.1f}m {speed:.1f}km/h"
            cv2.putText(frame, label_text, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for det in last_sign_detections:
            sx1, sy1, sx2, sy2 = map(int, det["bbox"])
            c_id = det["class_id"]
            conf = det["conf"]
            
            label = SIGN_CLASSES.get(c_id, f"Sign_{c_id}")
            text = f"{label} {conf*100:.0f}%"

            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 0, 255), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (sx1, sy1 - 20), (sx1 + tw, sy1), (255, 0, 255), -1)
            cv2.putText(frame, text, (sx1, sy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("ADAS Pilot - Full Integration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()