import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "fresh_rotten_cnn.keras"
IMG_SIZE = (224, 224)

st.set_page_config(
    page_title="Fresh vs Rotten Detector",
    page_icon="🍎",
    layout="centered"
)

# ==========================================
# TITLE
# ==========================================

st.title("🍎 Fresh vs Rotten Fruit Detector")

st.write(
    "Use your camera to scan a fruit or upload an image "
    "to check whether it is Fresh or Rotten. "
    "Supported fruits: Apple, Banana, Mango, Orange, and Strawberry only."
)

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

# ==========================================
# INPUT METHOD
# ==========================================

st.subheader("Choose how you want to scan")

input_method = st.radio(
    "Select an option:",
    ["📷 Scan with Camera", "📁 Upload Image"],
    horizontal=True
)

image = None

# ==========================================
# CAMERA
# ==========================================

if input_method == "📷 Scan with Camera":

    st.info("📷 Place the fruit in front of your camera and capture a photo.")

    camera_image = st.camera_input("Take a picture of your fruit")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")

# ==========================================
# UPLOAD
# ==========================================

else:

    uploaded_file = st.file_uploader(
        "Upload a fruit image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")


# ==========================================
# PREDICTION
# ==========================================

if image is not None:

    st.divider()

    st.subheader("🖼️ Fruit Image")

    st.image(
        image,
        caption="Image to be analyzed",
        width="stretch"
    )

    # --------------------------------------
    # PREPROCESS IMAGE
    # --------------------------------------

    processed_image = image.resize(IMG_SIZE)

    image_array = np.array(processed_image)

    image_array = image_array.astype("float32") / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------
    # MODEL PREDICTION
    # --------------------------------------

    with st.spinner("🤖 AI is analyzing the fruit..."):

        prediction = model.predict(
            image_array,
            verbose=0
        )

    rotten_probability = float(prediction[0][0])

    fresh_probability = 1 - rotten_probability

    # --------------------------------------
    # CLASSIFICATION
    # --------------------------------------

    if rotten_probability >= 0.5:

        predicted_class = "Rotten"
        confidence = rotten_probability * 100

    else:

        predicted_class = "Fresh"
        confidence = fresh_probability * 100

    # ======================================
    # RESULT
    # ======================================

    st.divider()

    st.subheader("🔍 AI Result")

    if predicted_class == "Fresh":

        st.success(
            f"🍎 FRESH\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    else:

        st.error(
            f"⚠️ ROTTEN\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    # ======================================
    # PROBABILITY
    # ======================================

    st.subheader("📊 Prediction Probability")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🍎 Fresh",
            f"{fresh_probability * 100:.2f}%"
        )

    with col2:

        st.metric(
            "⚠️ Rotten",
            f"{rotten_probability * 100:.2f}%"
        )

    # ======================================
    # MODEL DETAILS
    # ======================================

    with st.expander("🤖 View Model Details"):

        st.write("**Model:** CNN")

        st.write("**Input Size:** 224 × 224")

        st.write("**Classes:** Fresh / Rotten")

        st.write("**Activation:** Sigmoid")

        st.write(
            f"Fresh probability: "
            f"{fresh_probability:.4f}"
        )

        st.write(
            f"Rotten probability: "
            f"{rotten_probability:.4f}"
        )

else:

    st.divider()

    st.info(
        "👆 Choose Camera or Upload an image to start the analysis."
    )