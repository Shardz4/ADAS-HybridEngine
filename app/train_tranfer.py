import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

def main():
    print("--- ADAS Transfer Learning Pipeline (MobileNetV3) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using compute device: {device}")

    data_dir = r"C:\Users\CREWMOBILE\Videos\raw_lights"

    # ==========================================
    # 1. HOSTILE DATA AUGMENTATION 
    # ==========================================
    # We artificially create glare, dark nights, and bumps to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((64, 32)), 
        transforms.ColorJitter(brightness=0.6, contrast=0.5, saturation=0.5), # Simulate night/glare
        transforms.RandomRotation(10), # Simulate bumpy roads
        transforms.ToTensor(),
        # ImageNet standardization requirements
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"[INFO] Loaded {len(dataset)} images. Augmentation active.")

    # ==========================================
    # 2. THE TRANSFER LEARNING SURGERY
    # ==========================================
    print("[INFO] Downloading pre-trained ImageNet Foundation Model...")
    # Load the highly optimized MobileNetV3-Small with pre-trained weights
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Freeze the early layers so we don't destroy its understanding of shapes and edges
    for param in model.features.parameters():
        param.requires_grad = False

    # Perform surgery: Replace the final 1000-class ImageNet layer with our 4-class layer
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, 4)
    
    model = model.to(device)

    # We only optimize the newly attached classifier head
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.002)

    # ==========================================
    # 3. FINE-TUNING LOOP
    # ==========================================
    epochs = 10
    print(f"\n[INFO] Fine-tuning the custom head for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        accuracy = 100 * correct / total
        print(f"Epoch [{epoch+1:02d}/{epochs}] - Loss: {running_loss/len(dataloader):.4f} - Accuracy: {accuracy:.2f}%")

    # ==========================================
    # 4. EXPORT TO ONNX
    # ==========================================
    print("\n[INFO] Surgery and Fine-tuning complete! Exporting to ONNX...")
    model.eval()
    
    dummy_input = torch.randn(1, 3, 64, 32, device=device)
    onnx_path = "../models/traffic_lights_transfer.onnx"
    
    os.makedirs("../models", exist_ok=True)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    print(f"✅ Production model exported to: {onnx_path}")

if __name__ == "__main__":
    main()