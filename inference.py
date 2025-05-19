# inference.py

import torch
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
from PIL import Image
import requests
import os

# Define the number of classes
NUM_CLASSES = 4

# Define class names in the correct order
class_names = ['Anthracnose', 'Bacterial Canker', 'Powdery Mildew', 'Healthy']

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Download model weights if not already present
model_url = "https://huggingface.co/spaces/yashaswia/mango-disease-detector/resolve/main/vit_mango_disease.pth"
model_path = "vit_mango_disease.pth"

if not os.path.exists(model_path):
    print("🔽 Downloading model weights...")
    r = requests.get(model_url)
    with open(model_path, "wb") as f:
        f.write(r.content)

# Load model architecture and weights
model = vit_b_16(pretrained=False)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, NUM_CLASSES)
state_dict = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state_dict)
model.eval()

# Prediction function
def predict(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
        return class_names[predicted.item()]
    except Exception as e:
        raise RuntimeError(f"Failed to transform or predict image: {str(e)}")
