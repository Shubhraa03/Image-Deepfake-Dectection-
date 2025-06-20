import torch
import torch.nn as nn
import torch.optim as optim
import os

from models.cnn import CNNModel
from utils.dataset_loader import get_data_loaders

print("started...")
def main():
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("Training started...")

    # Config
    DATA_DIR = "data/deepfake"  # Replace with actual dataset folder
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Load Data
    print("Loading dataset...")
    train_loader, val_loader, idx_to_class = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)

   # train_loader, val_loader = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    print("Dataset loaded successfully.")
    print(f"Training batches: {len(train_loader)}")

    # Initialize model
    model = CNNModel().to(device)

    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS} started")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                print(f"Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        train_accuracy = 100 * correct / total
        train_loss = running_loss

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total

        # Summary of epoch
        print(f"Epoch {epoch + 1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/cnn_model.pth")
    print("Model saved as cnn_model.pth")

if __name__ == "__main__":
    main()