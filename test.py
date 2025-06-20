# test.py

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)
import seaborn as sns
import random
import time  # NEW
from datetime import datetime  # NEW

# =======================
# Step 1: Setup & Device
# =======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# =======================
# Step 2: Load Trained Model
# =======================
# Assuming CNNModel is defined in models/cnn.py
from models.cnn import CNNModel

model = CNNModel().to(device)
model.load_state_dict(torch.load("saved_models/cnn_model.pth", map_location=device))
model.eval()
print("[INFO] CNN model loaded successfully.")

# =======================
# Step 3: Load Test Data
# =======================
test_dir = "data/deepfake/test"  # Ensure test images are inside this folder with class-wise subfolders

transform = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
class_names = test_dataset.classes
print(
    f"[INFO] Found {len(test_dataset)} test images across {len(class_names)} classes: {class_names}"
)

# =======================
# Step 4: Prediction Loop
# =======================
all_preds = []
all_labels = []
all_probs = []  # NEW: List to store probabilities for ROC AUC and Log Loss

start_time = time.time()  # NEW: Start timer
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # Get predictions
        _, preds = torch.max(outputs, 1)

        # NEW: Get probabilities for the positive class (assuming class '1' is positive)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy()) # Store probabilities for all classes for Log Loss

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

testing_time = time.time() - start_time  # NEW: Calculate total testing time
print("[INFO] Predictions completed.")
print(f"[INFO] Testing finished in {testing_time:.2f} seconds.")


# =========================================================
# NEW: Step 5: Calculate Core Metrics & Save to Excel
# =========================================================
# Note: 'fit_time' is not applicable as we are only testing. Using 'testing_time'.
accuracy = accuracy_score(all_labels, all_preds)
error_rate = 1 - accuracy
# Assuming binary classification or using weighted average for multi-class
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
f1 = f1_score(all_labels, all_preds, average='weighted')

# For ROC AUC in binary case, we need probabilities of the positive class
# For multi-class, it requires one-vs-rest or one-vs-one approach
if len(class_names) == 2:
    # Extract probabilities for the positive class (class 1)
    probs_positive_class = np.array(all_probs)[:, 1]
    roc_auc = roc_auc_score(all_labels, probs_positive_class)
else: # For multi-class
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')

logloss = log_loss(all_labels, all_probs)

# Create a dictionary to hold the metrics
metrics_dict = {
    "accuracy": [accuracy],
    "error_rate": [error_rate],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1],
    "roc_auc": [roc_auc],
    "log_loss": [logloss],
    "fit_time": [testing_time], # Using testing time as fit_time
}

# Create a DataFrame and save to a uniquely named Excel file
df_metrics = pd.DataFrame(metrics_dict)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
excel_filename = f"results/model_performance_{timestamp}.xlsx"

os.makedirs("results", exist_ok=True) # Ensure the directory exists
df_metrics.to_excel(excel_filename, index=False)
print(f"[INFO] Core performance metrics saved to {excel_filename}")


# =======================
# Step 6: Classification Report
# =======================
report = classification_report(
    all_labels, all_preds, target_names=class_names, output_dict=True
)
df_report = pd.DataFrame(report).transpose()
df_report.to_csv("results/classification_report.csv")
print("[INFO] Classification report saved to results/classification_report.csv")

# =======================
# Step 7: Confusion Matrix
# =======================
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues"
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()
print("[INFO] Confusion matrix saved to results/confusion_matrix.png")

# =======================
# Step 8: Save Predictions to CSV
# =======================
image_paths = [path for path, _ in test_dataset.samples]
predicted_labels = [class_names[p] for p in all_preds]
actual_labels = [class_names[l] for l in all_labels]

df_predictions = pd.DataFrame(
    {"Image Path": image_paths, "Actual Label": actual_labels, "Predicted Label": predicted_labels}
)
df_predictions.to_csv("results/predictions.csv", index=False)
print("[INFO] Predictions saved to results/predictions.csv")

# =======================
# Step 9: Show Sample Predictions (Images with Labels)
# =======================
def show_sample_predictions():
    indices = random.sample(range(len(test_dataset)), min(6, len(test_dataset)))
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4) # Adjust spacing
    for ax, idx in zip(axs.flatten(), indices):
        image, label = test_dataset[idx]
        image_input = image.unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(image_input)
            _, pred = torch.max(output, 1)
        image = image.permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(
            f"Actual: {class_names[label]}\nPredicted: {class_names[pred.item()]}"
        )
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("results/sample_predictions.png")
    plt.close()
    print("[INFO] Sample prediction image saved to results/sample_predictions.png")

show_sample_predictions()

# =======================
# Done!
# =======================
print(
    "[SUCCESS] Testing complete. All results saved inside 'results/' folder."
)