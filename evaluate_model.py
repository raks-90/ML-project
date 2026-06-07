import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model
import cv2
import os

# Load trained CNN model
model = load_model("models/fingerprint_model.h5")

# Dataset path
DATASET_PATH = "dataset_clean"

IMG_SIZE = (128, 128)

y_true = []
y_score = []

# Preprocessing function
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    return np.expand_dims(img, axis=0)

print("Generating ROC curve...")

# Loop through dataset
for label in os.listdir(DATASET_PATH):
    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        try:
            img = preprocess_image(img_path)

            # Prediction
            pred = model.predict(img, verbose=0)[0]

            # Probability of "REAL" class (index 1)
            y_score.append(float(pred[1]))

            # Convert label to binary (REAL=1, FAKE=0)
            y_true.append(1 if label.lower() == "real" else 0)

        except Exception as e:
            print(f"Skipping {img_path}: {e}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], '--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Fingerprint Spoof Detection")
plt.legend()

# ✅ Save for frontend
plt.savefig("static/roc_curve.png")

print("ROC curve saved to static/roc_curve.png")
print(f"AUC Score: {roc_auc:.4f}")