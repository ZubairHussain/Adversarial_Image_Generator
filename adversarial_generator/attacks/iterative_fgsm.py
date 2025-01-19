import torch

def iterative_fgsm_attack(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    epsilon: float,
    alpha: float,
    num_iterations: int,
) -> torch.Tensor:
    """
    Perform iterative FGSM attack to generate adversarial noise.
    Args:
        model (torch.nn.Module): Pre-trained model.
        input_tensor (torch.Tensor): Input image tensor.
        target_class (int): Desired target class.
        epsilon (float): Maximum perturbation magnitude.
        alpha (float): Step size for each iteration.
        num_iterations (int): Maximum number of iterations.
    Returns:
        torch.Tensor: Adversarial image tensor.
    """
    # Clone the input tensor to ensure the original is not modified
    adv_tensor = input_tensor.clone().detach().requires_grad_(True)
    for i in range(num_iterations):
        # Forward pass
        output = model(adv_tensor)
        loss = torch.nn.CrossEntropyLoss()(output, torch.tensor([target_class], device=input_tensor.device))

        # Backward pass to compute gradients
        model.zero_grad()
        loss.backward()

        # Generate adversarial perturbation
        gradient = adv_tensor.grad.data
        adv_tensor = adv_tensor + alpha * gradient.sign()
        adv_tensor = torch.clamp(adv_tensor, input_tensor - epsilon, input_tensor + epsilon)  # Clip perturbation
        adv_tensor = torch.clamp(adv_tensor, 0, 1)  # Ensure pixel values are valid

        # Check if the target class is achieved
        predicted_class = output.argmax(dim=1).item()
        if predicted_class == target_class:
            print(f"Target class {target_class} achieved at iteration {i+1}")
            break
        
        # Retain gradients for the next iteration
        adv_tensor = adv_tensor.detach().requires_grad_(True)

    return adv_tensor