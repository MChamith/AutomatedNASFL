import torch
from torch import nn
import torch.nn.functional as F

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(AdvancedCNN, self).__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=False)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.resnet(x)
        return x