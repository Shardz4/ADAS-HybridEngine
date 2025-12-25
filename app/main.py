import cv2
import numpy as np
import time
import adas_pilot  # Compiled Rust library
from ultralytics import YOLO

def main():

    print("Loading YOLOv8 model..")
    model = YOLO("assets/models/yolov8n.pt")


    tracker = adas_pilot.RustTracker()
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
        current_time = time.time()
        dt = current_time - prev_time

        # Call Rust for lane detection
        try:
            lines = adas_pilot.detect_lanes_rust(frame)
            if lines.shape[0] >= 2:  # Expect at least 2 lines (left, right)
                for i in range(min(2, lines.shape[0])):
                    x1, y1, x2, y2 = map(int, lines[i])
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 10)
        except Exception as e:
            pass
        
        # Object Detection
        results = model(frame, verbose=False, classes =[2,3,5,7]) # classes: Car, bike, Bus, Truck
        detections_for_rust =[]

        for result in results:
            for box in result.boxes:
                x1,y1,x2,y2 = map(float, box.xyxy[0])
                w = x2-x1
                h = y2-y1
                detections_for_rust.append((x1,y1,w,h))

    #Tracking and collisontime calculation
    #raw boxes to rst and obtain physics data
    #tracked objects = [id, x, y, w, h , dist, speed, collisiontime)
    
    tracked_obj = tracker.process_frame(detections_for_rust, dt)

    #overlays

    for obj in tracked_obj:
        oid, x, y, w, h, dist, speed, collisiontime = obj

        #warning system
        #Green: Safe
        color = (0,255,0)
        status = "SAFE"
        thickness = 2

        #Yellow: Caution

        if collisiontime < 5.0 or dist< 20.0:
            color = (0,255,255)
            status = "CAUTION"
            thickness = 2

        #Red: Danger

        if collisiontime < 2.5:
            color - (0,0,255)
            status = "DANGER"
            thickness = 4

        cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), color, thickness)

        cv2.rectangle(frame, (int(x), int(y)-20), (int(x) + 120, int(y)), color, -1)

        cv2.putText(frame, f"{status}", (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0),2)

        label = f"{dist:.1f}m {speed:.1f}km/h {collisiontime:.1f}s"

        if collisiontime > 50:
            label = f"{dist:.1f}m {speed:.1f}km/h --.-s"
        cv2.putText(frame, label, (int(x), int(y+h)+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('ADAS Pilot - Week 2: Lane + object', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):  # 30 FPS
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()