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
    
    if st.button("Know more") :
        if predicted_class == "Melanoma":
            st.title("MELANOMA :")
            st.header("Cause:")
            st.write("Caused by UV radiation (sun exposure/tanning beds) damaging melanocytes (pigment-producing cells), often due to genetic mutations.")
            st.header("Effects:")
            st.write("Aggressive skin cancer that can metastasize (spread) if untreated, leading to life-threatening complications.")
            st.header("Precautions/Cure:")
            st.write("Early surgical removal is key. Use sunscreen, avoid excessive sun exposure, and monitor moles (ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving).")

        elif predicted_class == "Melanocytic Nevus":
            st.title("MELANOCYTIC NEVUS (MOLE):")
            st.header("Cause:")
            st.write("Benign proliferation of melanocytes, often due to genetics or sun exposure.")
            st.header("Effects:")
            st.write("Usually harmless but may rarely turn into melanoma (if dysplastic or changing).")
            st.header("Precautions/Cure:")
            st.write("Monitor for changes (size, color, shape). Suspicious moles should be biopsied/excised.")
        
        elif predicted_class == "Basal Cell Carcinoma":
            st.title("BASAL CELL CARCINOMA (BCC):")
            st.header("Cause:")
            st.write("UV-induced mutations in basal cells (most common skin cancer).")
            st.header("Effects:")
            st.write("Slow-growing, locally invasive but rarely metastatic. Can cause disfigurement if untreated.")
            st.header("Precautions/Cure:")
            st.write("Surgical removal, cryotherapy, or topical treatments. Prevent with sun protection (hats, sunscreen).")
        
        elif predicted_class == "Actinic Keratosis":
            st.title("ACTINIC KERATOSIS (AK):")
            st.header("Cause:")
            st.write("Chronic sun exposure damaging keratinocytes (precancerous).")
            st.header("Effects:")
            st.write("Rough, scaly patches; may progress to squamous cell carcinoma (SCC).")
            st.header("Precautions/Cure:")
            st.write("Treated with cryotherapy, topical creams (5-FU, imiquimod), or photodynamic therapy. Sun avoidance is critical.")
        
        elif predicted_class == "Benign Keratosis":
            st.title("BENIGN KERATOSIS (SEBORRHEIC KERATOSIS):")
            st.header("Cause:")
            st.write("Unknown, but linked to aging and genetics (not UV-related).")
            st.header("Effects:")
            st.write("Waxy, raised, non-cancerous lesions; no health risk but may be itchy or cosmetically bothersome.")
            st.header("Precautions/Cure:")
            st.write("Removal optional (cryotherapy, scraping). No prevention needed.")
        
        elif predicted_class == "Dermatofibroma":
            st.title("DERMATOFIBROMA:")
            st.header("Cause:")
            st.write("Benign fibrous tumor, often from minor trauma (e.g., insect bite).")
            st.header("Effects:")
            st.write("Firm, pigmented nodule; harmless but may be tender.")
            st.header("Precautions/Cure:")
            st.write("Usually left untreated; excision if symptomatic.")
        
        elif predicted_class == "Vascular Lesion":
            st.title("VASCULAR LESION (e.g., Hemangioma, Angioma):")
            st.header("Cause:")
            st.write("Abnormal blood vessel growth (congenital or acquired).")
            st.header("Effects:")
            st.write("Red/purple marks; may bleed if traumatized (rarely serious).")
            st.header("Precautions/Cure:")
            st.write("Laser therapy for cosmetic concerns. Monitor for changes.")
        
        else:
            st.warning("Unknown condition. Please consult a dermatologist for further evaluation.")
        
