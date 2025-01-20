
# Adversarial Image Generator

A Python library for generating adversarial images using various attack methods, including FGSM, Iterative FGSM, PGD, and Carlini & Wagner (C&W). The library supports PyTorch models and uses the Adversarial Robustness Toolbox (ART) for certain attack methods.

---

## Features
- **FGSM Attack**: Fast Gradient Sign Method.
- **Iterative FGSM**: Iterative version of FGSM for stronger perturbations.
- **C&W Attack**: Carlini & Wagner L2 attack for precise, optimization-based perturbations.

---

## Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/ZubairHussain/Adversarial_Image_Generator.git
cd adversarial-image-generator
```

### **2. Install Dependencies**
Install the required dependencies using pip:
```bash
pip install .
```

Alternatively, use `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

### **Command-Line Interface (CLI)**
You can generate adversarial images directly from the command line using the installed package.

#### **Example Commands**
1. **FGSM Attack**:
   ```bash
   adversarial_generator --input input.jpg --output adv_fgsm.jpg --target_class 281 --attack_type fgsm --epsilon 0.01
   ```

2. **Iterative FGSM Attack**:
   ```bash
   adversarial_generator --input input.jpg --output adv_iterative.jpg --target_class 9 --attack_type iterative_fgsm --epsilon 0.03 --alpha 0.005 --num_iterations 10
   ```

4. **Carlini & Wagner Attack**:
   ```bash
   adversarial_generator --input input.jpg --output adv_cw.jpg --target_class 207 --attack_type cw --max_iter 1000 --learning_rate 0.01
   ```

---

### **Library Usage**
You can also use the library programmatically.

#### **Example**
```python
from adversarial_generator.generator import AdversarialImageGenerator

# Initialize generator
generator = AdversarialImageGenerator(
    model_name="resnet50", epsilon=0.01, alpha=0.005, num_iterations=10
)

# Generate adversarial image
generator.generate(
    image_path="input.jpg",
    target_class=281,  # Tabby Cat
    attack_type="fgsm",
    output_path="adv_image.jpg"
)
```

---

## Supported Attack Types
| **Attack Type**  | **Description**                                         | **Parameters**                                                |
|-------------------|---------------------------------------------------------|---------------------------------------------------------------|
| `fgsm`           | Fast Gradient Sign Method                                | `epsilon` (magnitude of perturbation)                         |
| `iterative_fgsm` | Iterative version of FGSM                                | `epsilon`, `alpha` (step size), `num_iterations`              |
| `cw`             | Carlini & Wagner L2 attack                               | `max_iter`, `learning_rate`                                   |

---

## Examples
### **Visualizing Adversarial Examples**
```python
import matplotlib.pyplot as plt
from PIL import Image

# Load images
original = Image.open("input.jpg")
adversarial = Image.open("adv_image.jpg")

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(adversarial)
axes[1].set_title("Adversarial Image")
axes[1].axis("off")

plt.show()
```

---

## Notes:

1. The FGSM and iterative-FGSM attacks doesn't always generates adversarial noise that leads the model to misclassifying to the desired target class.
2. C&W attack is more effective in generating adversarial noise that leads to the desired predicted target class but it is computationally very expensive.
3. For further improvements:
    1. Tune and optimized hyperparameters.
    2. Explore other effective adversarial attacks.
