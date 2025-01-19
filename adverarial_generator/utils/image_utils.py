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
    

def postprocess_image(tensor: torch.Tensor) -> Image.Image:
    """
    Convert the adversarial image tensor back to a PIL image.
    Args:
        tensor (torch.Tensor): Image tensor.
    Returns:
        Image.Image: PIL Image.
    """
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
        transforms.ToPILImage(),
    ])
    return inv_transform(tensor.squeeze())