# streamlit_app.py

import streamlit as st
from PIL import Image
from inference import predict  # 👈 Make sure this is correct

st.set_page_config(page_title="Mango Leaf Disease Detector", layout="centered")

st.title("🍃 Mango Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a mango leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    image.save("temp.jpg", format="JPEG")

    with st.spinner("🔎 Predicting..."):
        try:
            result = predict("temp.jpg")
            st.success(f"🩺 Prediction: **{result}**")
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
