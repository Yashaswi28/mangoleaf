from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms

def predict(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        raise ValueError(f"Cannot identify or open image file: {image_path}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # Adjust based on your model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        input_tensor = transform(image).unsqueeze(0)
    except Exception as e:
        raise RuntimeError(f"Failed to transform image: {str(e)}")

    # === Run your model prediction here ===
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()

    return predicted_class
