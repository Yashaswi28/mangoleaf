import torch
from torchvision import transforms
from PIL import Image

import requests
import traceback
import streamlit as st

def predict(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)

        return "Test Passed"

    except Exception as e:
        full_trace = traceback.format_exc()
        st.error("An error occurred while transforming the image.")
        st.text(full_trace)  # Show the error on the Streamlit UI
        raise RuntimeError(f"Failed to transform image: {str(e)}")
