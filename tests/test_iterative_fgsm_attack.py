import pytest
import os
from PIL import Image
import torch
from adversarial_generator.generator import AdversarialImageGenerator
from adversarial_generator.utils.image_utils import preprocess_image


def test_fgsm_attack():

    input_image_path = 'data/ImageNet_samples/n02124075_Egyptian_cat.jpeg'
    output_image_path = 'output/iterative_fgsm_attack.png'
    target_class = 282 #  Tiger cat
    epsilon = 0.05
    alpha = 0.005
    num_iterations = 10

    # Initialize the generator
    generator = AdversarialImageGenerator(model_name="resnet50", epsilon=epsilon, alpha=alpha, num_iterations=num_iterations)

    # Generate the adversarial image
    adv_tensor = generator.generate(
        image_path=input_image_path,
        target_class=target_class,
        attack_type='iterative-fgsm',
        output_path=output_image_path,
    )

    # Check if output file exists
    assert os.path.exists(output_image_path), "Output adversarial image not saved"

    # Predict the class of the adversarial image
    predicted_class = generator.predict(preprocess_image(output_image_path))

    # Assert that the predicted class matches the target class
    assert predicted_class == target_class, (
        f"Adversarial image was classified as {predicted_class}, "
        f"but the target class was {target_class}"
    )