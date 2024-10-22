import torch
from torch import nn
import torch.nn.functional as F

class ComplexCNN2(nn.Module):
    def __init__(self):
        super(ComplexCNN2, self).__init__()
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(64 * 32 * 32, 100)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 64 * 32 * 32)
        x = self.fc(x)
        return x