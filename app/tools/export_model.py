from ultralytics import YOLO

# 1. Load your trained model
model_path = r"C:\Users\CREWMOBILE\Desktop\ADAS\adas_pilot\app\adas_training\indian_signs_v1\weights\best.pt"
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

# 2. Export to ONNX
print("Exporting to ONNX...")
model.export(format='onnx', dynamic=False)
print("Success! Your .onnx file is in the 'weights' folder.")
