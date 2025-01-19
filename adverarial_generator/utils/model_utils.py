import torchvision.models as models

def load_model(model_name: str = "resnet50"):
    """
    Load a pre-trained model from torchvision.
    Args:
        model_name (str): Name of the model to load.
    Returns:
        torch.nn.Module: Pre-trained model.
    """
    try:
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Model {model_name} is not supported")
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")