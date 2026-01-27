import os
import torch
from ultralytics import YOLO, settings

settings.update({'sync': False})

def train():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Hardware: {device.upper()}")

    dataset_yaml = r"C:\Users\CREWMOBILE\Desktop\ADAS\adas_pilot\assets\datasets\data.yaml"

    if not os.path.exists(dataset_yaml):
        print(f"Error: {dataset_yaml} not found.")
        return

    model = YOLO('yolov8n.pt') 

    model.train(
        data=dataset_yaml, 
        epochs=30, 
        imgsz=640, 
        device=device,
        project='adas_training',
        name='indian_signs_v1',
        batch=8,
        patience=5,
        workers=0,
        exist_ok=True,
        save=True,      
        save_period=5    
    )

    print("Exporting to ONNX...")
    model.export(format='onnx', dynamic=False)

if __name__ == '__main__':
    train()