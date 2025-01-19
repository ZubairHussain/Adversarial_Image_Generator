import argparse
from adversarial_generator.generator import AdversarialImageGenerator

def main():
    """
    Main function to execute the adversarial image generation pipeline.
    """

    parser = argparse.ArgumentParser(description="Adversarial Image Generator using FGSM")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the adversarial image",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        required=True,
        help="Target class index for the adversarial attack",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Magnitude of the adversarial perturbation (default: 0.01)",
    )

    args = parser.parse_args()

    try:
        # Initialize the adversarial image generator
        generator = AdversarialImageGenerator(epsilon=args.epsilon)

        # Generate the adversarial image
        generator.generate(
            image_path=args.input,
            target_class=args.target_class,
            output_path=args.output
        )
        print(f"Adversarial image saved at: {args.output}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()