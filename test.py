import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog

IMG_SIZE = 128

# Load model
model = load_model("models/fingerprint_model.h5")

# File picker
Tk().withdraw()

img_path = filedialog.askopenfilename(
    title="Select Fingerprint Image",
    filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
)

if img_path == "":
    print("❌ No image selected")
    exit()

# Preprocess (RGB)
img = cv2.imread(img_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img / 255.0
img = img.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# Predict
prediction = model.predict(img)

# Output
if prediction[0][0] > 0.5:
    print("❌ FAKE FINGERPRINT")
else:
    print("✅ REAL FINGERPRINT")

input("\nPress Enter to exit...")