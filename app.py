# app.py
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# -------------------------------
# Streamlit page config
# -------------------------------
st.set_page_config(page_title="Medical Image Classifier", layout="centered")
st.title("Medical Image Classification App")

# -------------------------------
# Define class labels
# -------------------------------
CLASS_LABELS = ["Normal", "Pneumonia"]

# -------------------------------
# Determine model path safely
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'my_model.h5')

st.write("Looking for model at:", MODEL_PATH)
st.write("Exists?", os.path.exists(MODEL_PATH))

# -------------------------------
# Load model with caching
# -------------------------------
@st.cache_resource
def load_model_cached(path):
    if not os.path.exists(path):
        st.error(f"Model not found at {path}")
        return None
    model = load_model(path, compile=False)
    return model

model = load_model_cached(MODEL_PATH)

if model is None:
    st.stop()  

# -------------------------------
# Print model input shape
# -------------------------------
st.write("Model input shape:", model.input_shape)

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    temp_file_path = os.path.join(BASE_DIR, "temp_image.jpg")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)
    
    # -------------------------------
    # Preprocess image automatically
    # -------------------------------
    input_shape = model.input_shape  # e.g., (None, height, width, channels)
    target_height, target_width = input_shape[1], input_shape[2]
    channels = input_shape[3]

    color_mode = 'rgb' if channels==3 else 'grayscale'

    img = image.load_img(temp_file_path, target_size=(target_height, target_width), color_mode=color_mode)
    img_array = image.img_to_array(img)

    if channels==1 and img_array.shape[-1]!=1:
        img_array = np.expand_dims(img_array[:,:,0], axis=-1)

    img_array = np.expand_dims(img_array, axis=0)  
    img_array /= 255.0 

    # -------------------------------
    # Make prediction
    # -------------------------------
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.success(f"Predicted class: {predicted_label} ({confidence:.2f}% confidence)")
