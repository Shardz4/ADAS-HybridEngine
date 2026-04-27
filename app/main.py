import cv2
import numpy as np
import time
import os 
import threading
import queue

# AI Libraries
import torch
import onnxruntime as ort
from ultralytics import YOLO

# Your Custom Rust Engine
import adas_pilot

# ==========================================
# CONFIGURATION
# ==========================================
IS_TWO_WAY_ROAD = False  
AI_SKIP_FRAMES = 3  # Run heavy AI every 3rd frame

# Map your Indian traffic sign classes here
SIGN_CLASSES = {
    52: "STOP",
    # 0: "SPEED_LIMIT_30",
    # 1: "SPEED_LIMIT_50",
}

# ==========================================
# MULTI-THREADED CAMERA PIPELINE
# ==========================================
class ThreadedCamera:
    """Runs the video decoding and resizing on a separate CPU core to unblock the main thread."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.q = queue.Queue(maxsize=5) 
        self.stopped = False
        
        self.t = threading.Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.t.start()
        return self

    def update(self):
        while not self.stopped:
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                
                # Resize on the background thread to save CPU cycles on the main thread
                frame = cv2.resize(frame, (1280, 720))
                self.q.put(frame)
            else:
                time.sleep(0.001) 

    def read(self):
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
    print("--- Starting ADAS Master Suite (Hardware Accelerated) ---")
    
    # ------------------------------------------
    # 1. LOAD PYTORCH YOLO MODEL & WARMUP
    # ------------------------------------------
    print("[DEBUG] Loading YOLOv8 model for vehicles...")
    try:
        model = YOLO('yolov8n.pt')
        model.to('cuda') # Force PyTorch to use the NVIDIA GPU
        
        print("[DEBUG] Warming up GPU VRAM...")
        # Send a dummy frame to force VRAM allocation before the video starts
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        model(dummy_frame, verbose=False)
        print("[DEBUG] Vehicle Model loaded and VRAM Locked!")
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO to GPU: {e}")
        return

    # ------------------------------------------
    # 2. LOAD RUST SUBSYSTEMS & ONNX
    # ------------------------------------------
    print("[DEBUG] Initializing Rust modules...")
    try:
        tracker = adas_pilot.Tracker()
        manager = adas_pilot.LaneManager(smoothing=0.6, is_two_way=IS_TWO_WAY_ROAD)
        brain = adas_pilot.AdasBrain("../models/traffic_signs.onnx") 
        print(f"[DEBUG] ONNX Execution Providers: {ort.get_available_providers()}")
        print("[DEBUG] Rust modules and ONNX Brain initialized.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize Rust modules: {e}")
        return

    # ------------------------------------------
    # 3. START VIDEO STREAM
    # ------------------------------------------
    video_path = "../assets/videos/test_vid.mp4" 
    if not os.path.exists(video_path):
        video_path = "test_video.mp4" # Fallback to local directory

    print(f"[DEBUG] Opening multi-threaded stream: {video_path}")
    stream = ThreadedCamera(video_path).start()
    time.sleep(1.0) # Give the buffer a second to fill

    if not stream.more():
        print("[ERROR] Failed to open the video file.")
        return

    print("[DEBUG] Streaming live... Press 'q' to quit.")
    
    # State variables
    prev_time = time.time()
    frame_count = 0
    active_left, active_right = None, None
    light_status = "NONE"
    last_sign_detections = []
    last_raw_yolo_detections = []

    # ------------------------------------------
    # 4. MAIN INFERENCE LOOP
    # ------------------------------------------
    while stream.more():
        frame = stream.read()
        frame_count += 1
        h, w, _ = frame.shape
        
        cur_time = time.time()
        dt = cur_time - prev_time
        prev_time = cur_time
        if dt <= 0: dt = 0.033

        # --- MEDIUM SUBSYSTEMS (Run Every 2nd Frame) ---
        if frame_count % 2 == 0:
            light_status = adas_pilot.check_traffic_lights(frame)
            try:
                raw_lines_np = adas_pilot.detect_lanes(frame)
                raw_lines_list = [tuple(x) for x in raw_lines_np]
                l_tup, r_tup = manager.update_lanes(raw_lines_list, float(w))
                if l_tup != (0.,0.,0.,0.): active_left = l_tup
                if r_tup != (0.,0.,0.,0.): active_right = r_tup
            except Exception:
                pass # Suppress lane warnings in production

        # --- HEAVY AI SUBSYSTEMS (Run Every N Frames) ---
        if frame_count % 2 == 0:
            try:
                raw_lines_np = adas_pilot.detect_lanes(frame)
                raw_lines_list = [tuple(x) for x in raw_lines_np]
                l_tup, r_tup = manager.update_lanes(raw_lines_list, float(w))
                if l_tup != (0.,0.,0.,0.): active_left = l_tup
                if r_tup != (0.,0.,0.,0.): active_right = r_tup
            except:
                pass
            
            # B. Vehicles (PyTorch YOLO)
            results = model(frame, verbose=False, classes=[2, 3, 5, 7])
            last_raw_yolo_detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    
                    # --- ADD THIS EGO MASK ---
                    # If the bottom of the bounding box (y2) is in the lowest 100 pixels 
                    # of the screen, it's our own hood. Ignore it!
                    if y2 > (h - 100):
                        continue
                    # -------------------------
                    
                    last_raw_yolo_detections.append((x1, y1, x2-x1, y2-y1))

        # --- TRACKER (Run Every Frame) ---
        filtered_data = manager.filter_objects(last_raw_yolo_detections)
        bboxes_only = []
        ego_map = {} 
        for (bbox, is_ego) in filtered_data:
            bboxes_only.append(bbox)
            ego_map[int(bbox[0])] = is_ego

        tracked_objs = tracker.process_frame(bboxes_only, dt)

        # ------------------------------------------
        # 5. VISUALIZATION HUD
        # ------------------------------------------
        
        # Draw Lanes
        if active_left:
            c = (0,255,0) if IS_TWO_WAY_ROAD else (255,0,0)
            cv2.line(frame, (int(active_left[0]), int(active_left[1])), (int(active_left[2]), int(active_left[3])), c, 3)
        if active_right:
            cv2.line(frame, (int(active_right[0]), int(active_right[1])), (int(active_right[2]), int(active_right[3])), (255,0,0), 3)
        
        # Draw Traffic Lights
        light_color = (255, 255, 255) 
        if light_status == "RED": light_color = (0,0,255)
        elif light_status == "YELLOW": light_color = (0,255,255)
        elif light_status == "GREEN": light_color = (0,255,0)

        if light_status != "NONE":
            cv2.rectangle(frame, (20,20) , (250,80), (0,0,0), -1)
            cv2.putText(frame, f"LIGHT: {light_status}", (30,65), cv2.FONT_HERSHEY_SIMPLEX, 1.2, light_color, 3)

        # Draw Tracked Vehicles
        for obj in tracked_objs:
            oid, x, y, bw, bh, dist, speed, ttc = obj
            is_in_ego_lane = ego_map.get(int(x), False)

            color = (0,255,0)
            if dist < 25.0 or (ttc < 5.0 and ttc > 0): color = (0 , 255, 255)
            if is_in_ego_lane and (ttc < 2.5 and ttc > 0): color = (0,0,255)
            
            cv2.rectangle(frame, (int(x), int(y)), (int(x+bw), int(y+bh)), color, 2)
            label_text = f"{dist:.1f}m {speed:.1f}km/h"
            cv2.putText(frame, label_text, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw Traffic Signs
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

        # Draw FPS Counter
        fps = 1.0 / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Render
        cv2.imshow("ADAS Pilot - Full Integration", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    stream.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()