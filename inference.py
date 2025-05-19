import traceback
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
import numpy as np
import streamlit as st
import os
import requests
model_path = "vit_mango_disease.pth"
model_url = "https://huggingface.co/spaces/yashaswia/mango-disease-detector/resolve/main/vit_mango_disease.pth"

# Download if not already present
if not os.path.exists(model_path):
    print("Downloading model...")
    r = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(r.content)
    print("Model downloaded.")

# Load model
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()


# Define the same class names used during training
class_names = ["Anthracnose", "Bacterial Canker", "Cutting Weevil", "Gall Midge", "Healthy", "Powdery Mildew", "Sooty Mould"]

# Define transformation (match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict(image_path):
    try:
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            class_idx = predicted.item()
            return f"🩺 Prediction: {class_names[class_idx]}"
        
    except Exception as e:
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        raise RuntimeError(f"Failed to transform image: {str(e)}")
