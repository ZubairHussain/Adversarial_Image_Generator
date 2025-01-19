import torch

def fgsm_attack(model: torch.nn.Module, input_tensor: torch.Tensor, target_class: int, epsilon: float) -> torch.Tensor:
    """
    Perform FGSM attack to generate adversarial noise.
    Args:
        model (torch.nn.Module): Pre-trained model.
        input_tensor (torch.Tensor): Input image tensor.
        target_class (int): Desired target class.
        epsilon (float): Magnitude of the noise.
    Returns:
        torch.Tensor: Adversarial image tensor.
    """
    input_tensor.requires_grad = True
    try:
        output = model(input_tensor)
        loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target_class], device=input_tensor.device))
        model.zero_grad()
        loss.backward()

        # Generate adversarial noise
        gradient = input_tensor.grad.data
        adv_tensor = input_tensor + epsilon * gradient.sign()
        return torch.clamp(adv_tensor, 0, 1)
    except Exception as e:
        raise RuntimeError(f"Error during FGSM attack: {e}")