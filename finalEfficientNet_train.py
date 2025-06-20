import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

# ---------------- Configuration ----------------
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = "data/deepfake"
MODEL_SAVE_PATH = "saved_models/final_efficientNet_model.pth"

# Create model save directory
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# ---------------- Data Loaders ----------------
def get_data_loaders(data_dir, batch_size):
    print("\n[INFO] Loading and transforming data...")

    transform_train = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "validation")

    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("[INFO] Classes:", train_dataset.classes)
    print("[INFO] Class mapping:", train_dataset.class_to_idx)
    print("[INFO] Training samples:", len(train_dataset))
    print("[INFO] Validation samples:", len(val_dataset))

    return train_loader, val_loader, train_dataset.classes

# ---------------- Model Definition ----------------
def build_model(num_classes):
    print("\n[INFO] Building EfficientNet model...")
    model = models.efficientnet_b0(pretrained=True)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)

# ---------------- Training Function ----------------
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, scheduler):
    print("\n[INFO] Starting training...")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloader, desc=f"{phase.upper()} Epoch {epoch+1}/{num_epochs}"):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"[INFO] Best model saved with accuracy: {best_acc:.4f}")

    print("\n[INFO] Training complete.")
    print(f"[INFO] Best validation accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# ---------------- Main Execution ----------------
def main():
    train_loader, val_loader, class_names = get_data_loaders(DATA_DIR, BATCH_SIZE)
    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    trained_model = train_model(model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer, scheduler)
    print(f"\n[INFO] Final model saved to {MODEL_SAVE_PATH}")

# ---------------- Safe Entry Point ----------------
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
