# import tensorflow as tf
# from tensorflow.keras import layers, models
# import os

# # Dataset path
# DATASET_PATH = "dataset_clean"

# # Image size and batch size
# IMG_SIZE = (128, 128)
# BATCH_SIZE = 32

# # Load dataset
# train_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_PATH,
#     validation_split=0.2,
#     subset="training",
#     seed=123,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# val_ds = tf.keras.utils.image_dataset_from_directory(
#     DATASET_PATH,
#     validation_split=0.2,
#     subset="validation",
#     seed=123,
#     image_size=IMG_SIZE,
#     batch_size=BATCH_SIZE
# )

# # Normalize data
# normalization_layer = layers.Rescaling(1./255)
# train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# # Build CNN model
# model = models.Sequential([
#     layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
#     layers.MaxPooling2D(),
    
#     layers.Conv2D(64, (3,3), activation='relu'),
#     layers.MaxPooling2D(),
    
#     layers.Conv2D(128, (3,3), activation='relu'),
#     layers.MaxPooling2D(),
    
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(2, activation='softmax')  # real / fake
# ])

# # Compile model
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )

# # Train model
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=10
# )

# # Save model
# model.save("models/fingerprint_model.h5")


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

DATASET_PATH = "dataset_clean"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

print("Classes:", train_ds.class_names)

# Normalize
normalization_layer = layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Improve Performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# Data Augmentation
# data_augmentation = tf.keras.Sequential([
#     layers.RandomFlip("horizontal"),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.20),
    layers.RandomZoom(0.20),
    layers.RandomContrast(0.20),
    layers.RandomTranslation(0.10, 0.10),
])


# CNN Model
model = models.Sequential([

    data_augmentation,

    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(128,128,3)),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(256, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(2, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early Stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=25,
    callbacks=[early_stop]
)

# Save
model.save("models/fingerprint_model.h5")

print("Model Saved Successfully")