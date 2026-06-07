import tensorflow as tf

DATASET_PATH = "dataset_clean"

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128,128),
    batch_size=32
)

print("Classes:", train_ds.class_names)