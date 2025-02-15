from PIL import Image
import torch
from typing import Optional, Union
from .utils.image_utils import preprocess_image, postprocess_image
from .utils.model_utils import load_model
from .attacks.fgsm import fgsm_attack
from .attacks.iterative_fgsm import iterative_fgsm_attack
from .attacks.cw import cw_attack


class AdversarialImageGenerator:
    def __init__(self, model_name: str = "resnet50", epsilon: float = 0.01, alpha: float = 0.005, learning_rate: float = 0.01, num_iterations: int = 10):
        """
        Initialize the adversarial image generator.
        Args:
            model_name (str): The name of the pre-trained model to use (default: "resnet50").
            epsilon (float): The magnitude of the perturbation (default: 0.01).
        """
        if epsilon <= 0 or epsilon > 1:
            raise ValueError("Epsilon must be a positive value between 0 and 1")
        self.model = load_model(model_name)
        self.epsilon = epsilon
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def generate(self, image_path: str, target_class: int, attack_type: str = "fgsm", output_path: Optional[str] = None) -> Union[Image.Image, None]:
        """
        Generate an adversarial image.
        Args:
            image_path (str): Path to the input image.
            target_class (int): Desired target class index for the adversarial image.
            output_path (Optional[str]): Path to save the adversarial image (default: None).
        Returns:
            Union[Image.Image, None]: The adversarial image as a PIL Image object if output_path is None.
        """
        try:
            # Step 1: Preprocess the input image
            input_tensor = preprocess_image(image_path)

            # Step 2: Generate the adversarial image tensor using FGSM
            if attack_type == "fgsm":
                adv_tensor = fgsm_attack(self.model, input_tensor, target_class, self.epsilon)
            elif attack_type == "iterative-fgsm":
                adv_tensor = iterative_fgsm_attack(self.model, input_tensor, target_class, self.epsilon, self.alpha, self.num_iterations)
            elif attack_type == "cw":
                adv_tensor = cw_attack(self.model, input_tensor=input_tensor, target_class=target_class, max_iter=self.num_iterations, learning_rate=self.learning_rate)
            else:
                raise ValueError(f"Unsupported attack type: {attack_type}")

            # Step 3: Postprocess the adversarial image tensor into a PIL image
            adv_image = postprocess_image(adv_tensor)

            # Step 4: Save or return the adversarial image
            if output_path:
                adv_image.save(output_path)
                print(f"Adversarial image saved at: {output_path}")
                return None
            else:
                return adv_image

        except FileNotFoundError:
            raise FileNotFoundError(f"Input image file not found at: {image_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to generate adversarial image: {e}")
        
    def predict(self, input_tensor: torch.Tensor) -> int:
        """
        Predict the class for input tensor.
        Args:
            input_tensor (torch.Tensor): Input tensor
        Returns:
            int: Predicted class index.
        """
        try:
            predicted_class = self.model(input_tensor).argmax(dim=1).item()
        except Exception as e:
            raise RuntimeError(f"Failed to predict class : {e}")
        return predicted_class
        