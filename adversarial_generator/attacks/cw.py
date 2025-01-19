import torch
from art.attacks.evasion import CarliniL2Method
from art.estimators.classification import PyTorchClassifier

def cw_attack(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    max_iter: int = 1000,
    learning_rate: float = 0.01,
) -> torch.Tensor:
    """
    Perform Carlini & Wagner (C&W) L2 attack to generate adversarial noise.
    Args:
        model (torch.nn.Module): Pre-trained PyTorch model.
        input_tensor (torch.Tensor): Input image tensor.
        target_class (int): Desired target class.
        max_iter (int): Maximum number of iterations for optimization (default: 1000).
        learning_rate (float): Learning rate for the optimizer (default: 0.01).
    Returns:
        torch.Tensor: Adversarial image tensor.
    """
    # Wrap the PyTorch model using ART's PyTorchClassifier
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0, 1),
        loss=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate),
        input_shape=(3, 224, 224),
        nb_classes=1000,
    )

    # One-hot encode the target class
    target_class_one_hot = torch.zeros((1, 1000))
    target_class_one_hot[0, target_class] = 1

    # Convert input tensor to NumPy
    input_numpy = input_tensor.detach().cpu().numpy()

    # Create C&W attack instance
    cw_attack = CarliniL2Method(classifier=classifier, targeted=True, max_iter=max_iter)

    # Generate adversarial example
    adv_numpy = cw_attack.generate(x=input_numpy, y=target_class_one_hot.numpy())

    # Convert back to PyTorch tensor
    adv_tensor = torch.from_numpy(adv_numpy).to(input_tensor.device)

    return adv_tensor