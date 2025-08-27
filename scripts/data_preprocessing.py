# scripts/data_preprocessing.py
import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_mnist(digits=[4,2], num_samples=10):
    """
    Load MNIST dataset, filter for specific digits, flatten and convert to binary (-1, +1)
    digits: list of digits to store (default [0,1])
    num_samples: number of samples per digit
    Returns:
        patterns: numpy array of shape (M, 784) with values -1 or +1
    """
    # Load MNIST
    (x_train, y_train), (_, _) = mnist.load_data()
    
    patterns = []
    
    for d in digits:
        # Ensure we don't try to pick more samples than are available
        digit_images = x_train[y_train.astype(int) == d][:num_samples]
        for img in digit_images:
            img_flat = img.flatten()  # flatten 28x28 -> 784
            # convert to binary (-1, +1)
            img_binary = np.where(img_flat > 127, 1, -1)
            patterns.append(img_binary)
    
    patterns = np.array(patterns)
    return patterns

if __name__ == "__main__":
    patterns = load_and_preprocess_mnist()
    print("Patterns shape:", patterns.shape)
    print("First pattern example:\n", patterns[0])
