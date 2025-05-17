from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

def predict(image_path):
    try:
        image = Image.open(image_path)
        print(f"Image mode: {image.mode}, size: {image.size}, format: {image.format}")
        image = image.convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify or open image file: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        input_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        print(f"Transform failed: {str(e)}")  # DEBUG
        raise RuntimeError(f"Failed to transform image: {str(e)}")

    # Dummy model prediction for test
    return "Dummy Result"
