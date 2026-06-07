import os
import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(img_path):
    img = cv2.imread(img_path)  # RGB image
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img

def load_data(dataset_path):
    data = []
    labels = []

    categories = ["real", "fake"]

    for label, category in enumerate(categories):
        folder = os.path.join(dataset_path, category)

        for img_name in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img_name)
                img = preprocess_image(img_path)

                data.append(img)
                labels.append(label)
            except:
                pass

    data = np.array(data).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    labels = np.array(labels)

    return data, labels