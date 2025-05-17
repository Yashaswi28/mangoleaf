
import torch
from torchvision import transforms
from PIL import Image

def load_model():
    model = torch.load("vit_mango_disease.pth", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

def predict(img_path):
    image = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    class_names = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']
    return class_names[predicted.item()]
