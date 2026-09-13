import os
import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


# ==========================================
# 1. BASIC SETTINGS
# ==========================================

DATASET_DIR = "Fruits"
MODEL_PATH = "fresh_rotten_cnn.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("TensorFlow version:", tf.__version__)


# ==========================================
# 2. FIND ALL IMAGES AND CREATE LABELS
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


print("\nTotal images:", len(images))
print("Fresh images:", labels.count(0))
print("Rotten images:", labels.count(1))


# ==========================================
# 3. RECREATE THE SAME TRAIN/VAL/TEST SPLIT
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

    # Convert pixel values from 0-255 to 0-1
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


# ==========================================
# 5. CREATE TEST DATASET
# ==========================================

test_ds = tf.data.Dataset.from_tensor_slices(
    (test_images, test_labels)
)


# ==========================================
# 6. APPLY PREPROCESSING
# ==========================================

test_ds = test_ds.map(
    load_and_preprocess,
    num_parallel_calls=AUTOTUNE
)


# ==========================================
# 7. BATCH TEST DATA
# ==========================================

test_ds = (
    test_ds
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)


print("\nTest dataset created successfully!")


# ==========================================
# 8. LOAD SAVED MODEL
# ==========================================

print("\nLoading saved model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 9. BASIC KERAS EVALUATION
# ==========================================

print("\n==========================================")
print("MODEL EVALUATION")
print("==========================================")

test_loss, test_accuracy = model.evaluate(test_ds)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ==========================================
# 10. GET PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

predicted_probabilities = model.predict(test_ds)

# Convert probabilities to binary labels
# Probability >= 0.5 → Rotten (1)
# Probability < 0.5 → Fresh (0)

predicted_labels = (
    predicted_probabilities >= 0.5
).astype(int).flatten()

actual_labels = np.array(test_labels)


print("Predictions generated successfully!")


# ==========================================
# 11. CALCULATE METRICS
# ==========================================

accuracy = accuracy_score(
    actual_labels,
    predicted_labels
)

precision = precision_score(
    actual_labels,
    predicted_labels
)

recall = recall_score(
    actual_labels,
    predicted_labels
)

f1 = f1_score(
    actual_labels,
    predicted_labels
)


# ==========================================
# 12. PRINT METRICS
# ==========================================

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("==========================================")


# ==========================================
# 13. CLASSIFICATION REPORT
# ==========================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        actual_labels,
        predicted_labels,
        target_names=["Fresh", "Rotten"]
    )
)


# ==========================================
# 14. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    actual_labels,
    predicted_labels
)

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

print(cm)


# ==========================================
# 15. DISPLAY CONFUSION MATRIX
# ==========================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Fresh", "Rotten"]
)

disp.plot()

plt.title("Fresh vs Rotten - Confusion Matrix")
plt.show()


# ==========================================
# 16. FINISHED
# ==========================================

print("\n==========================================")
print("EVALUATION COMPLETED")
print("==========================================")