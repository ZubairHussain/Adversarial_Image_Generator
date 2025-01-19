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
        "--attack_type", 
        type=str, 
        default="fgsm", 
        choices=["fgsm", "iterative_fgsm", "cw"], 
        help="Type of attack")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.01,
        help="Magnitude of the adversarial perturbation (default: 0.01)",
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.001, 
        help="Step size for iterative attacks (default: 0.001)")
    parser.add_argument(
        "--num_iterations", 
        type=int, 
        default=10, 
        help="Number of iterations for iterative attacks")
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=0.01, 
        help="Learning rate for CW attack")
    

    args = parser.parse_args()

    try:
        # Initialize the adversarial image generator
        generator = AdversarialImageGenerator(epsilon=args.epsilon, alpha=args.alpha, learning_rate=args.learning_rate, num_iterations=args.num_iterations)

        # Generate the adversarial image
        generator.generate(
            image_path=args.input,
            target_class=args.target_class,
            attack_type=args.attack_type,
            output_path=args.output
        )
        print(f"Adversarial image saved at: {args.output}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()