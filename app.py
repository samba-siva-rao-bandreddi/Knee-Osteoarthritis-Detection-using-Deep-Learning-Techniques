import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
from PIL import Image


# Load the trained model
@st.cache_resource
def load_model_custom():
    model = load_model(r"2nd model.h5")  # Modify path
    return model


model = load_model_custom()

# Get the expected input shape
input_shape = model.input_shape[1:]  # Exclude batch dimension
EXPECTED_SIZE = (input_shape[0], input_shape[1])  # (height, width)
EXPECTED_CHANNELS = input_shape[-1]  # Number of channels (1 = grayscale, 3 = RGB)

# Define the class labels
CLASS_LABELS = ["Normal", "Doubtful", "Mild","Moderate","Severe"]


# Define Image Preprocessing
def preprocess_image(img):
    img = img.resize(EXPECTED_SIZE)  # Resize image to match model input shape
    img = np.array(img)

    if EXPECTED_CHANNELS == 1:  # Convert to grayscale if the model expects 1 channel
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.expand_dims(img, axis=-1)  # Add channel dimension

    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize pixel values
    return img


# Streamlit UI
st.title("🦵 Knee Osteoarthritis Detection")
st.write("Upload an X-ray image to classify the knee condition.")

# Upload Image
uploaded_file = st.file_uploader("Upload a Knee X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    #st.image(image, caption="Uploaded Image", use_container_width=True)  # Fixed deprecation warning

    # Preprocess Image
    processed_img = preprocess_image(image)

    # Model Prediction
    prediction = model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class] * 100

    # Create two columns
    col1, col2 = st.columns([1, 1])  # Adjust the ratio as needed

    with col1:
        # Display the uploaded image with a specific width
        st.image(image, caption="Uploaded Image", width=300)  # Adjust width as needed

    with col2:
        # Display class probabilities
        st.write("### Class Probabilities:")
        for i, label in enumerate(CLASS_LABELS):
            st.write(f"{label}: {prediction[0][i] * 100:.2f}%")

    # Display Prediction Result
    st.write("### Prediction Result")
    st.success(f"Predicted Class: **{CLASS_LABELS[predicted_class]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Additional Insights
    st.write("### Interpretation:")
    if predicted_class == 0:
        st.info("✅ The knee appears **Normal** with no signs of osteoarthritis.")
    elif predicted_class == 1:
        st.warning("⚠️ The knee shows **Doubtful** signs of osteoarthritis.")
    elif predicted_class == 2:
        st.warning("⚠️ The knee shows **Mild** osteoarthritis.")
    elif predicted_class == 3:
        st.warning("⚠️ The knee shows **Moderate** osteoarthritis.")
    else:
        st.error("🚨 The knee shows **Severe** osteoarthritis, medical attention advised!")
