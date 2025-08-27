# scripts/main.py
import numpy as np
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_mnist
from hopfield_network import HopfieldNetwork
import os # Import the os module

def add_noise(pattern, noise_level=0.0):
    """
    Flip some pixels randomly to add noise
    pattern: input vector (-1/+1)
    noise_level: fraction of pixels to flip
    """
    noisy = pattern.copy()
    num_flip = int(len(pattern) * noise_level)
    flip_indices = np.random.choice(len(pattern), num_flip, replace=False)
    noisy[flip_indices] *= -1  # flip the pixel
    return noisy

def plot_patterns(original, noisy, recalled):
    """
    Show original, noisy, and recalled images side by side
    and save the result to a file.
    """
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    
    for ax, img, title in zip(axes, [original, noisy, recalled],
                              ['Original', 'Noisy', 'Recalled']):
        # Reshape the flattened 784-pixel vector back to a 28x28 image
        ax.imshow(img.reshape(28,28), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Check if the 'results' directory exists, and create it if it doesn't
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the figure to the specified file path
    output_filepath = os.path.join(output_dir, "results.png")
    plt.savefig(output_filepath)
    print(f"\nImage saved successfully to: {output_filepath}")

if __name__ == "__main__":
    # Step 1: Load and preprocess MNIST
    # We will store only a few patterns to stay within the network's capacity.
    digits_to_store = [1,3]  # Reduced to two distinct digits
    patterns = load_and_preprocess_mnist(digits=digits_to_store, num_samples=1)
    print("Loaded patterns shape:", patterns.shape)
    
    # Step 2: Train Hopfield Network
    hopfield_net = HopfieldNetwork()
    hopfield_net.train(patterns)
    print("Network trained with", patterns.shape[0], "patterns")
    
    # Step 3: Pick a test image and add noise
    test_idx = np.random.randint(0, patterns.shape[0])
    original = patterns[test_idx]
    
    # Note: A low noise level (e.g., 0.1) is often necessary for successful recall.
    # The more noise, the harder it is for the network to find the correct minimum energy state.
    noisy = add_noise(original, noise_level=0.4)
    
    # Step 4: Recall memory
    recalled = hopfield_net.recall(noisy, max_steps=100,visualize_energy=True)
    
    # Step 5: Visualize
    plot_patterns(original, noisy, recalled)

    # Final result check
    is_correct = np.array_equal(recalled, original)
    print("\n--- Result ---")
    print(f"Original pattern retrieved successfully: {is_correct}")
