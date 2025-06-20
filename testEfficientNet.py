import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    log_loss, precision_score, recall_score, f1_score, accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
DEVICE = torch.device("cpu")
BATCH_SIZE = 16
MODEL_PATH = "saved_models/final_efficientNet_model.pth"
TEST_DIR = "data/deepfake/Test"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -----------------------------
# PREPROCESSING
# -----------------------------
print("[INFO] Initializing test data...")
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_dataset = ImageFolder(TEST_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
class_names = test_dataset.classes
print(f"[INFO] Classes found: {class_names}")

# -----------------------------
# MODEL LOADING
# -----------------------------
print("[INFO] Loading trained EfficientNet model...")
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -----------------------------
# INFERENCE
# -----------------------------
print("[INFO] Starting testing...")
start_time = time.time()
y_true, y_pred, y_scores = [], [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs[:, 1].cpu().numpy())  # confidence for class 'Real'

# -----------------------------
# METRICS CALCULATION
# -----------------------------
print("[INFO] Calculating performance metrics...")
acc = accuracy_score(y_true, y_pred)
err = 1 - acc
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc = roc_auc_score(y_true, y_scores)
loss = log_loss(y_true, y_scores)
fit_time = round(time.time() - start_time, 2)

# -----------------------------
# SAVE CLASSIFICATION REPORT
# -----------------------------
print("[INFO] Saving classification report...")
report = classification_report(
    y_true, y_pred, target_names=class_names, output_dict=True
)
report_df = pd.DataFrame(report).transpose()
report_df.to_excel(
    os.path.join(RESULTS_DIR, "efficientNetFinalMatrix_classification_report.xlsx")
)

# -----------------------------
# SAVE CONFUSION MATRIX PNG
# -----------------------------
print("[INFO] Saving confusion matrix image...")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix - EfficientNet")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "efficientNetFinalMatrix_confusion_matrix.png"))
plt.close()

# -----------------------------
# SAVE PERFORMANCE SUMMARY
# -----------------------------
print("[INFO] Saving model performance summary...")
performance = {
    "accuracy": [acc],
    "error_rate": [err],
    "precision": [prec],
    "recall": [rec],
    "f1_score": [f1],
    "roc_auc": [roc],
    "log_loss": [loss],
    "fit_time": [fit_time]
}
perf_df = pd.DataFrame(performance)
perf_df.to_excel(
    os.path.join(RESULTS_DIR, "efficientNetFinalMatrix_performance_summary.xlsx"),
    index=False
)

# -----------------------------
# DONE
# -----------------------------
print("\n✅ Testing complete! All results saved in the 'results/' folder.")
