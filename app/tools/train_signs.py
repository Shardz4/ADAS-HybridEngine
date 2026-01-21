from ultralytics import YOLO
import torch
import os

def train():

    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    dataset_yaml= r"C:\Users\CREWMOBILE\Desktop\ADAS\adas_pilot\assets\datasets\data.yaml"

    if not os.path,exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset YAML file not found at {dataset_yaml}")

        model = YOLO('yolov8n.pt')

        model.train(
            data=dataset_yaml,
            epochs = 30,
            imgsz = 640,
            device = device,
            project = r'C:\Users\CREWMOBILE\Desktop\ADAS\adas_pilot\models',
            name ='indian_signs_v1',
            batch = 8,
            patienece = 5,
            save = True,
            save_period = 5
        )

        print("Exporting to ONNX format...")
        model.export(format = 'onnx', dynamic = false)
        


