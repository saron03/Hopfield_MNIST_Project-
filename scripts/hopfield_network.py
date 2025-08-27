# scripts/hopfield_network.py
import os
import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self):
        self.weights = None  # weight matrix will be stored here

    def train(self, patterns):
        """
        Store patterns in the network using Hebb's rule
        patterns: numpy array of shape (M, N) where M = number of patterns, N = number of neurons
        """
        num_patterns, num_neurons = patterns.shape
        self.weights = np.zeros((num_neurons, num_neurons))

        # Hebb's rule: wij = (1/N) * sum over p of xi_p * xj_p
        # Normalize by number of neurons, not number of patterns
        for p in patterns:
            self.weights += np.outer(p, p)  # outer product of pattern with itself

        self.weights /= num_neurons

        # No self-connections
        np.fill_diagonal(self.weights, 0)

    def recall(self, pattern, max_steps=100,visualize_energy=False):
        """
        Recall a memory from a noisy pattern using random, asynchronous updates.
        The recall stops when the network state stabilizes.
        pattern: input vector (-1/+1)
        max_steps: maximum number of iterations
        visualize_energy: if True, track and plot energy over iterations
        """
        s = pattern.copy()
        num_neurons = len(s)
        
        energy_history = []
        if visualize_energy:
            energy_history.append(self.energy(s))
    
        for _ in range(max_steps):
            s_prev = s.copy()
            # Randomly shuffle the order of updates to prevent cycles
            permuted_indices = np.random.permutation(num_neurons)
            
            # Update all neurons in the random order
            for i in permuted_indices:
                net_input = np.dot(self.weights[i], s)
                s[i] = 1 if net_input >= 0 else -1
            
            if visualize_energy:
                 energy_history.append(self.energy(s))
            
            # Check for convergence - if the state hasn't changed, we've converged
            if np.array_equal(s, s_prev):
                print(f"Network converged in {_} steps.")
                if visualize_energy:
                    os.makedirs("results", exist_ok=True)
                    plt.figure(figsize=(6,4))
                    plt.plot(energy_history, marker='o')
                    plt.title("Hopfield Network Energy over Iterations")
                    plt.xlabel("Iteration")
                    plt.ylabel("Energy")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig("results/energy_plot.png")
                    plt.close()
                return s
        
        print("Network did not converge within the maximum steps.")
        if visualize_energy:
            os.makedirs("results", exist_ok=True)
            plt.figure(figsize=(6,4))
            plt.plot(energy_history, marker='o')
            plt.title("Hopfield Network Energy over Iterations")
            plt.xlabel("Iteration")
            plt.ylabel("Energy")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("results/energy_plot.png")
            plt.close()
        return s
        
    def energy(self, state):
        """
        Compute energy of the network for a given state
        """
        return -0.5 * np.dot(state, np.dot(self.weights, state))
