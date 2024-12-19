import torch

import torch.nn as nn
import torch.nn.functional as F

class TrafficSignModel(nn.Module):
    def __init__(self):
        super(TrafficSignModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout(p=0.15)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout(p=0.20)

        # After two conv + pool layers: 
        # Input: 30x30 -> Conv(5x5): 26x26 -> Conv(5x5):22x22 -> Pool(2x2):11x11
        # Conv(3x3):9x9 -> Conv(3x3):7x7 -> Pool(2x2):3x3
        # Final size: 256 * 3 * 3 = 2304 features
        self.fc1 = nn.Linear(256*3*3, 512)
        self.drop3 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(512, 43)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))    # 3 -> 32@26x26
        x = F.relu(self.conv2(x))    # 32 -> 64@22x22
        x = self.pool1(x)            # ->64@11x11
        x = self.drop1(x)
        
        x = F.relu(self.conv3(x))    # 64 ->128@9x9
        x = F.relu(self.conv4(x))    #128->256@7x7
        x = self.pool2(x)            # ->256@3x3
        x = self.drop2(x)

        x = x.view(x.size(0), -1)    # Flatten
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)              # raw logits
        return x