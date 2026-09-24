import torch.nn as nn
from torchvision import models

def get_model(num_classes=43):
    # Load pre-trained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the head for traffic signs
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model