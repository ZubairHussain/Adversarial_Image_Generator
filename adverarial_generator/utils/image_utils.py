from PIL import Image
import torch
from torchvision import transforms

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocess the input image for the model.
    Args:
        image_path (str): Path to the input image.
    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error while preprocessing image: {e}")