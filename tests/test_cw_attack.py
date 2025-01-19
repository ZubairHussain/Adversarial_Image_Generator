import pytest
import os
from PIL import Image
import torch
from adversarial_generator.generator import AdversarialImageGenerator
from adversarial_generator.utils.image_utils import preprocess_image


def test_cw_attack():

    input_image_path = 'data/ImageNet_samples/n02124075_Egyptian_cat.jpeg'
    output_image_path = 'output/CW_attack.png'
    target_class = 282 
    max_iter = 1000
    learning_rate = 0.01

    # Initialize the generator
    generator = AdversarialImageGenerator(model_name="resnet50", learning_rate=learning_rate, num_iterations=max_iter)

    # Generate the adversarial image
    adv_tensor = generator.generate(
        image_path=input_image_path,
        target_class=target_class,
        attack_type='cw',
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