import streamlit as st
from PIL import Image
from inference import predict  # make sure this function returns the predicted class
import base64
from io import BytesIO
import numpy as np


# === Set Page Config ===
st.set_page_config(
    page_title="Mango Leaf Disease Detector 🍃",
    layout="centered",
)

# === Custom Background Image with CSS ===
def set_bg():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://t3.ftcdn.net/jpg/05/61/96/06/360_F_561960690_uCMNRrqahIsdrOeEG7Lx5DzLPCof6GNe.jpg");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
            height: 100vh;       
        }}
        .title-text {{
            font-size: 36px;
            font-weight: bold;
            color: #ffffff;
            text-shadow: 1px 1px 2px #000000;
        }}
        .info-box {{
            background-color: rgba(255, 255, 255, 0.85);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        }}

        /* Make the image container full width */
        div[data-testid="stImage"] > img {{
            width: 100vw !important;       /* 100% viewport width */
            height: auto !important;       /* keep aspect ratio */
            margin-left: calc(-50vw + 50%);
        }}

        /* Remove horizontal padding around the image */
        .block-container {{
            padding-left: 0rem;
            padding-right: 0rem;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# === Apply background ===
set_bg()

# === App Title ===
st.markdown("<h1 class='title-text' align='center'>🍃 Mango Leaf Disease Detection App</h1>", unsafe_allow_html=True)
st.markdown("### Upload a mango leaf image to detect potential diseases.")

# === Upload File ===
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# === Prediction Section ===
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save the image properly as RGB
    image.save("temp.jpg", format="JPEG")

    with st.spinner("Predicting..."):
        result = predict("temp.jpg")


    # Show prediction result
    st.markdown(f"<div class='info-box'><h4>🩺 Prediction: <span style='color:#d62728;'>{result}</span></h4></div>", unsafe_allow_html=True)

