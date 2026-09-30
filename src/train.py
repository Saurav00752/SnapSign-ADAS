import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

class SnapSignNet(nn.Module):
    def __init__(self, num_classes=43):
        super(SnapSignNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 32 * 8 * 8)
        x = self.classifier(x)
        return x

def get_dataloaders(batch_size=64):
    # Resize all images to 32x32 to match the SnapSignNet architecture
    # Convert them to Tensors and normalize the pixel values
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Downloading and loading GTSRB dataset... (This may take a minute)")
    
    # PyTorch automatically downloads the dataset to a 'data' folder
    train_dataset = datasets.GTSRB(root='../data', split='train', download=True, transform=transform)
    test_dataset = datasets.GTSRB(root='../data', split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print(f"Successfully loaded {len(train_dataset)} training images and {len(test_dataset)} test images.")
    return train_loader, test_loader

def train_model(epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    model = SnapSignNet(num_classes=43).to(device)
    train_loader, test_loader = get_dataloaders()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Standard PyTorch Training Loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}")
                running_loss = 0.0

    print("Training complete!")
    
    # Save the model
    os.makedirs("../models", exist_ok=True)
    save_path = "../models/snapsign_base.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model architecture and weights saved to {save_path}")

if __name__ == "__main__":
    train_model(epochs=5)