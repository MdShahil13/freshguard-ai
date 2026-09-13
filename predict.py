import os
import tensorflow as tf
import numpy as np


# ==========================================
# 1. BASIC SETTINGS
# ==========================================

MODEL_PATH = "fresh_rotten_cnn.keras"

IMG_SIZE = (224, 224)


# ==========================================
# 2. LOAD SAVED MODEL
# ==========================================

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# 3. GET IMAGE PATH
# ==========================================

image_path = input("\nEnter image path: ").strip()

if not os.path.exists(image_path):
    print("Image not found!")
    exit()


# ==========================================
# 4. LOAD IMAGE
# ==========================================

image = tf.io.read_file(image_path)

image = tf.image.decode_image(
    image,
    channels=3,
    expand_animations=False
)

# Resize to the same size used during training
image = tf.image.resize(image, IMG_SIZE)

# Convert pixels from 0-255 to 0-1
image = tf.cast(image, tf.float32) / 255.0

# Add batch dimension
image = tf.expand_dims(image, axis=0)


# ==========================================
# 5. MAKE PREDICTION
# ==========================================

prediction = model.predict(image, verbose=0)

# Sigmoid output = probability of Rotten
rotten_probability = float(prediction[0][0])


# ==========================================
# 6. DETERMINE CLASS
# ==========================================

if rotten_probability >= 0.5:

    predicted_class = "Rotten"
    confidence = rotten_probability * 100

else:

    predicted_class = "Fresh"
    confidence = (1 - rotten_probability) * 100


# ==========================================
# 7. DISPLAY RESULT
# ==========================================

print("\n==========================================")
print("FRUIT CONDITION PREDICTION")
print("==========================================")

print("Image:", image_path)
print("Prediction:", predicted_class)
print(f"Confidence: {confidence:.2f}%")

print("==========================================")