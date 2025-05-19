# inference.py

import torch
import torchvision.transforms as transforms
from PIL import Image
import requests
import os

# Download model if not already present
model_url = "https://huggingface.co/spaces/yashaswia/mango-disease-detector/resolve/main/vit_mango_disease.pth"
model_path = "vit_mango_disease.pth"

if not os.path.exists(model_path):
    print("Downloading model...")
    r = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(r.content)

# Load model
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Define class names (update based on your actual dataset)
class_names = ['Anthracnose', 'Bacterial Canker', 'Powdery Mildew', 'Healthy']

def predict(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]
    except Exception as e:
        raise RuntimeError(f"Failed to transform or predict image: {str(e)}")
