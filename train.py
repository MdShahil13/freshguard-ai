import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ==========================================
# 1. BASIC SETTINGS
# ==========================================

DATASET_DIR = "Fruits"

MODEL_PATH = "fresh_rotten_cnn.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5

print("TensorFlow version:", tf.__version__)


# ==========================================
# 2. FIND IMAGES AND CREATE LABELS
# ==========================================

images = []
labels = []

for fruit in os.listdir(DATASET_DIR):

    fruit_path = os.path.join(DATASET_DIR, fruit)

    if not os.path.isdir(fruit_path):
        continue

    for category in ["Fresh", "Rotten"]:

        category_path = os.path.join(fruit_path, category)

        if not os.path.isdir(category_path):
            continue

        # Fresh = 0
        # Rotten = 1
        label = 0 if category == "Fresh" else 1

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            images.append(file_path)
            labels.append(label)


print("Total images:", len(images))
print("Fresh images:", labels.count(0))
print("Rotten images:", labels.count(1))


# ==========================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ==========================================

train_images, temp_images, train_labels, temp_labels = train_test_split(
    images,
    labels,
    test_size=0.20,
    random_state=42,
    stratify=labels
)


val_images, test_images, val_labels, test_labels = train_test_split(
    temp_images,
    temp_labels,
    test_size=0.50,
    random_state=42,
    stratify=temp_labels
)


print("\nDataset split:")
print("Training:", len(train_images))
print("Validation:", len(val_images))
print("Testing:", len(test_images))


# ==========================================
# 4. IMAGE PREPROCESSING FUNCTION
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess(image_path, label):

    # Read image from disk
    image = tf.io.read_file(image_path)

    # Decode image
    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    # Resize image
    image = tf.image.resize(image, IMG_SIZE)

    # Convert pixels from 0-255 to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ==========================================
# 5. CREATE TENSORFLOW DATASETS
# ==========================================

train_ds = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)
)

val_ds = tf.data.Dataset.from_tensor_slices(
    (val_images, val_labels)
)

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)
)


# ==========================================
# 6. APPLY PREPROCESSING
# ==========================================

train_ds = train_ds.map(
    load_and_preprocess,
    num_parallel_calls=AUTOTUNE
)

val_ds = val_ds.map(
    load_and_preprocess,
    num_parallel_calls=AUTOTUNE
)

test_ds = test_ds.map(
    load_and_preprocess,
    num_parallel_calls=AUTOTUNE
)


# ==========================================
# 7. BATCHING
# ==========================================

train_ds = (
    train_ds
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

val_ds = (
    val_ds
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    test_ds
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


print("\nTensorFlow datasets created successfully!")


# ==========================================
# 8. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])


# Apply augmentation only to training images
train_ds = train_ds.map(
    lambda image, label: (data_augmentation(image, training=True), label),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(AUTOTUNE)


print("Data augmentation added successfully!")

# ==========================================
# 9. BUILD CNN MODEL
# ==========================================

model = tf.keras.Sequential([

    # Input: 224 x 224 RGB image
    tf.keras.layers.Input(shape=(224, 224, 3)),

    # Block 1
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 2
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Block 3
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Convert feature maps to one-dimensional vector
    tf.keras.layers.Flatten(),

    # Fully connected layer
    tf.keras.layers.Dense(128, activation="relu"),

    # Reduce overfitting
    tf.keras.layers.Dropout(0.5),

    # Binary classification
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.summary()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)


model.save(MODEL_PATH)


print("\n==========================================")
print("MODEL TRAINING COMPLETED")
print("==========================================")
print("Model saved successfully!")
print("Saved as:", MODEL_PATH)
print("==========================================")