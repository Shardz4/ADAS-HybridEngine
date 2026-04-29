import cv2
import os
import uuid
from ultralytics import YOLO

def main():
    output_dir = "../dataset/raw_lights"
    os.markedirs(output_dir, exist_ok =True)
    print(f"[INFO] Output Directory Created: {output_dir}")

    model  =YOLO('yolov8n.pt')
    model.to('cuda')
    video_path = "../assets/videos/traffic.mp4"
    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_count = 0

    SKIP_FRAMES = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if frame_count % SKIP_FRAMES == 0:
            frame = cv2.resize(frame, (1200, 720))
            results = model(frame, verbose=False, classes=[9])

            for result in results:
                for box in result.boxes:
                    if float(box.conf[0]) > 0.40:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        light_crop = frame[y1:y2, x1:x2]

                        if light_crop.size > 0:
                            file_id = str(uuid.uuid4())[:8]
                            filename = os.path.join(output_dir, f"Light_{file_id}.jpg")
                            cv2.imwrite(filename, light_crop)
                            saved_count +=1
            cv2.imshow("Detections", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {saved_count} traffic light images to {output_dir}")

if __name__ == "__main__":
    main()

    

                        
            

