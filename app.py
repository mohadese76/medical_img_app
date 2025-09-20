
import streamlit as st
import pickle
import numpy as np

# 1️⃣ بارگذاری مدل
@st.cache_data  # برای کش کردن مدل و جلوگیری از بارگذاری دوباره
def load_model():
  model=tf.keras.models.load_model('./my-project/models/my_model.h5')
  return model

model = load_model()

# 2️⃣ عنوان اپلیکیشن
st.title("Test My Tensorflow Model")

# 3️⃣ گرفتن ورودی از کاربر
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)

# 4️⃣ دکمه پیش‌بینی
if st.button("Predict"):
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    prediction = model.predict(input_data)
    st.success(f"Prediction: {prediction}")