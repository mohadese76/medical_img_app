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
# Load model with caching
# -------------------------------
@st.cache_resource
def load_model_cached():
    model_path = './my-project/models/my_model.h5'  
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        return None
    model = load_model(model_path, compile=False)
    return model

model = load_model_cached()

if model is None:
    st.stop()  

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display uploaded image
    st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)
    
    # -------------------------------
    # Preprocess image
    # -------------------------------
    img = image.load_img(temp_file_path, target_size=(224, 224))  # اندازه مدل خودت
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # اضافه کردن بعد batch
    img_array /= 255.0  # نرمال‌سازی (اگر مدل شما نیاز دارد)
    
    # -------------------------------
    # Make prediction
    # -------------------------------
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = prediction[0][predicted_index] * 100
    
    st.success(f"Predicted class: {predicted_label} ({confidence:.2f}% confidence)")
