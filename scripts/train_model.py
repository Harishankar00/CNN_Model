# scripts/train_model.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from data_loader import CityscapesDataset, get_transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}\n")

# Dataset paths (train split; these directories contain subdirectories for each city)
image_dir = "/home/harishankar/CNN_Model/dataset/leftImg8bit_trainvaltest/leftImg8bit/train"
mask_dir = "/home/harishankar/CNN_Model/dataset/gtFine_trainvaltest/gtFine/train"

# Hyperparameters
batch_size = 8          # Adjust based on your GPU memory
num_classes = 20        # Adjust based on your dataset
learning_rate = 1e-4
num_epochs = 50
print(f"📌 Hyperparameters:\n - Batch size: {batch_size}\n - Learning rate: {learning_rate}\n - Num epochs: {num_epochs}\n")

# Load dataset
transform = get_transforms()
train_dataset = CityscapesDataset(image_dir, mask_dir, transform=transform, num_classes=num_classes)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

print(f"✅ Dataset Loaded: {len(train_dataset)} samples found.")

# Model setup: using DeepLabV3 with a ResNet-50 backbone (pretrained on COCO)
model = deeplabv3_resnet50(pretrained=True)
model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)  # Adjust final layer for segmentation
model = model.to(device)
print("✅ Model Loaded: DeepLabV3 (ResNet-50 Backbone) with modified classifier.\n")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore pixels marked as 255
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    print(f"\n🔹 Epoch [{epoch+1}/{num_epochs}] starting...\n")

    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)

        # Print input details for first batch
        if batch_idx == 0:
            print(f"🖼️ Sample Batch Shapes:\n - Images: {images.shape} (Batch x Channels x Height x Width)\n - Masks: {masks.shape} (Batch x Height x Width)")

        optimizer.zero_grad()
        outputs = model(images)['out']  # Get segmentation output (batch x num_classes x H x W)

        # Ensure mask dimensions match the output dimensions
        if outputs.shape[2:] != masks.shape[1:]:
            masks = torch.nn.functional.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode="nearest").squeeze(1).long()

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(train_loader) - 1:
            print(f"    🔄 [Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    print(f"\n✅ Epoch [{epoch+1}/{num_epochs}] completed. Avg Loss: {avg_loss:.4f}")

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/segmentation_model.pth")
print("\n🎯 Training complete. Model saved as 'models/segmentation_model.pth'.")
