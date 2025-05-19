import traceback
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import transforms
import numpy as np
def predict(image_path):
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = transform(image).unsqueeze(0)

        # Dummy return for testing
        return "Test Passed"

    except Exception as e:
        # Save full traceback
        with open("error_log.txt", "w") as f:
            f.write(traceback.format_exc())
        # Print it in console too
        print("=== FULL TRACEBACK ===")
        print(traceback.format_exc())

        raise RuntimeError(f"Failed to transform image: {str(e)}")
    # Return dummy output for now
    return "Test Passed"
    

