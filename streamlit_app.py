import streamlit as st
from PIL import Image
from inference import load_model, predict

st.title("🍃 Mango Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload a mango leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model()
    prediction = predict(image, model)

    st.success(f"Prediction: Class {prediction}")
