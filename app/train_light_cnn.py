import torch
import torch.nn as nn
import torch.nn.functional as f

class TrafficLightHead(nn.Module):
    def __init__(self):
        super(TrafficLightHead, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(64*4*8, 128)
        self.fc2 = nn.Linear(128, 4) 

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))

            x = x.view(-1, 64*4*8)
            x = F.relu(self.fc1(x))
            x =self.dropout(x)
            s = self.fc2(x)
            return x


if __name__ == "__main__":
    model =TrafficLightHead()
    dummy_crop = torch.randn(1, 3, 64, 32)
    output = model(dummy_crop)
    print(f"Model output: {output.shape}")