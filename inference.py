
import torch
from torchvision import transforms
from PIL import Image

import requests

def load_model():
    url = "https://huggingface.co/spaces/yashaswia/mango-disease-detector/blob/main/vit_mango_disease.pth"
    response = requests.get(url)
    with open("vit_mango_disease.pth", "wb") as f:
        f.write(response.content)

    model = torch.load("vit_mango_disease.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

def predict(image_path):
    # Define the transform (you can adjust size/mean/std as per your model's training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # or the size used during training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # standard ImageNet normalization
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # === Your model prediction code ===
    # Example:
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    # Return class name or label mapping if needed
    return predicted_class
