# Hopfield Network

This project implements a Hopfield network capable of memorizing and recalling handwritten digit patterns from the MNIST dataset. The network can take a distorted or noisy image as input and recover the original digit by converging to a stored memory. Using binary neuron states (-1, +1), Hebbian learning for pattern storage, and asynchronous updates for recall, the network demonstrates how associative memory works in practice. The workflow includes data preprocessing, network training, noise simulation, pattern recall, and visualization of results and network energy.

## Step 1: Data Preprocessing

In this step, we load and prepare the MNIST dataset for the Hopfield network.

### Load the MNIST dataset
- MNIST contains **28×28 grayscale images** of handwritten digits (0–9).  
- We use only the **training set**, since the Hopfield network is designed for **memorizing patterns**, not for generalizing to new data.

### Filter digits
- Only the digits specified (default `[1, 2]`) are kept.  
- This reduces the number of patterns to **stay within the Hopfield network’s capacity**.
- When you store too many digits, the network’s “memory valleys” start overlapping.
- A noisy input can then roll into the wrong valley or a spurious pattern.

### Flatten images
- Each image is **28×28 pixels**, which is flattened into a **784-element vector**.  
- Each neuron in the Hopfield network corresponds to **one pixel**.
- Flattening ensures the network can process each pixel individually.

### Convert to binary (-1, +1)
- Original pixel values are 0–255 (black to white).  
- Pixels above 127 are converted to **+1**, below 127 to **-1**.  
- This simplifies the data and makes it compatible with the **binary neuron states** of the Hopfield network.

### Output
- A **NumPy array** of shape `(num_patterns, 784)`.  
- Each row is a **single digit pattern**, ready to be stored in the Hopfield network.

---

## Step 2: Training and Recall (Hopfield Network)

In this step, we implement the **Hopfield network** and demonstrate how it can **memorize and recall patterns**, from **noisy inputs**.

---

### Initialize the Hopfield Network

* The network consists of **N neurons**, where **N = 784** for MNIST (one neuron per pixel).
* The weight matrix `weights` is initialized as a **zero matrix of shape N×N**.
* No self-connections are allowed; i.e., diagonal entries of the weight matrix are set to **0**.

---

### Train the network (Hebb's rule)

* Patterns prepared in Step 1 are stored using **Hebb’s learning rule**:

  $$
  w_{ij} = \frac{1}{N} \sum_{p} x_i^p x_j^p
  $$

  * `x_i^p` and `x_j^p` are the states of neurons `i` and `j` in pattern `p`.
* The weights store the **associations between pixels**, allowing the network to recall complete patterns from partial or noisy inputs.
* The more patterns you store, the higher the chance of **overlapping memory valleys**; careful selection of digits and samples is necessary to avoid spurious states.

---

### Recall a memory

* The network can recall a stored pattern from a **noisy or incomplete input**.
* **Asynchronous, random updates** are used:

  1. Each neuron is updated one at a time in **random order**.
  2. Neuron state is set to **+1 or -1** based on the weighted sum of inputs from other neurons.
  3. This process repeats until the network **converges** (no neuron changes its state).
* Convergence indicates the network has reached a **stable memory** (local energy minimum).

---

### Energy function (optional)

* Each network state has an associated **energy**:

  $$
  E(s) = -\frac{1}{2} s^T W s
  $$
* Energy decreases with each iteration of recall, moving the network toward a **stored memory valley**.
* This can be **visualized** to track the network’s convergence.

---

### Visualize energy over iterations

* The energy can be **tracked and plotted** to understand how the network “rolls downhill” toward a stored pattern.
* The plot can now be **saved automatically** to `results/energy_plot.png`.
* This shows a **monotonically decreasing curve**, confirming that the network is moving toward a stable memory.

---

### Input / Output

* **Input:**

  * A **pattern vector** (possibly noisy), shape `(784,)`.
  * Optionally, `visualize_energy=True` to track energy.
* **Output:**

  * The **recalled pattern** (shape `(784,)`), which should closely match a stored memory.
  * Saved **energy plot** if `visualize_energy=True`.

---

## Step 3: Pattern Recall and Visualization

In this step, we demonstrate how the Hopfield network can **recall stored patterns from noisy inputs** and visualize the results.

---

### 1. Add Noise to a Pattern

* Randomly flips a fraction of pixels in a stored pattern to simulate a **corrupted image**.
* `noise_level` specifies the fraction of pixels to flip (0 = no noise, 1 = all pixels flipped).
* Example:

```python
pattern = [1, -1, 1, -1]
noisy_pattern = add_noise(pattern, noise_level=0.5)
# Result: some pixels flipped randomly, e.g., [-1, -1, 1, 1]
```

---

### 2. Select Test Pattern

* Randomly choose one of the stored patterns as the **test input**.
* Adds noise to this pattern to create the **noisy input** for recall.

---

### 3. Recall Pattern

* Hopfield network tries to recover the **original pattern** from the noisy input.
* Uses **asynchronous, random updates** for neurons until the network **converges**.
* Optionally, track **energy over iterations**: the energy should decrease as the network stabilizes.
* Energy plot is saved automatically to `results/energy_plot.png`.

---

### 4. Visualize Results

* Display **three images side by side**:

  1. Original pattern
  2. Noisy input
  3. Recalled pattern
* Saves the visualization as `results/results.png`.

---

### 5. Check Recall Accuracy

* Compare the recalled pattern with the original:

```python
is_correct = np.array_equal(recalled, original)
```

* Returns `True` if the network **recalls the pattern exactly**, else `False`.
* High noise levels or storing too many patterns may result in **incorrect recall**.

---

### 6. Input / Output

**Input:**

* Stored patterns from Step 1
* Noise level for corrupting the test pattern

**Output:**

* Recalled pattern vector
* Saved visualization of original, noisy, and recalled patterns
* Energy plot showing convergence

---

### 7. Summary

This step demonstrates the **full Hopfield network workflow**:

1. Select and corrupt a test pattern.
2. Recall the pattern from noisy input.
3. Visualize the original, noisy, and recalled images.
4. Track network energy to confirm convergence.
5. Check if recall was successful.

---


