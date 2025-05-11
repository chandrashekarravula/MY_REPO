import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import os

# Set page title and layout
st.set_page_config(page_title="Skin Cancer Classification", layout="wide")

# Load the saved model
@st.cache_resource  # Cache model for faster reloads
def load_model():
    return tf.keras.models.load_model('skin_cancer_model.h5')  # or .keras

model = load_model()

# Define class labels (modify based on your dataset)
CLASS_NAMES = [
    "Melanoma",
    "Melanocytic Nevus",
    "Basal Cell Carcinoma",
    "Actinic Keratosis",
    "Benign Keratosis",
    "Dermatofibroma",
    "Vascular Lesion"
]

# Preprocess image for model input
def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    predicted_class = CLASS_NAMES[class_idx]
    return predicted_class, prediction

# Streamlit UI
st.title("Skin Cancer Classification using CNN 🏥")
st.write("Upload an image of a skin lesion for classification.")

# Add exit button in the sidebar
     # This will stop the script execution

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=300)

    # Preprocess and predict
    predicted_class, prediction = preprocess_image(image)
    # Show prediction
    st.subheader("Prediction Result")
    st.success(f"**Class:** {predicted_class}")

if st.button("Exit"):
    st.write("Exiting the application...")
    os._exit(1)
    SystemExit()

    