import torch
from torchvision import transforms
from PIL import Image
import requests

# Load model from Hugging Face or local
def load_model():
    url = "https://huggingface.co/spaces/yashaswia/mango-disease-detector/blob/main/vit_mango_disease.pth"
    response = requests.get(url)
    with open("vit_mango_disease.pth", "wb") as f:
        f.write(response.content)
    model = torch.load("vit_mango_disease.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = transform(image).unsqueeze(0)
    output = model(img)
    _, predicted = torch.max(output, 1)
    return predicted.item()
