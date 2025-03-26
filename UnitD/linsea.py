import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
M = 5  # Number of distinct elements
alphabet = ["A", "B", "C", "D", "E"]
search_key = "A"
n_values = [5, 10, 20, 50]
samples = 10000

# Store results
results = {}

for n in n_values:
    counts = {i: 0 for i in range(-1, n)}  # Includes -1 for "not found"

    # Generate random arrays and simulate linear search
    for _ in range(samples):
        arr = np.random.choice(alphabet, size=n, replace=True)
        try:
            idx = list(arr).index(search_key)
        except ValueError:
            idx = -1
        counts[idx] += 1

    # Empirical frequencies
    empirical_freqs = [counts[i] / samples for i in range(-1, n)]

    # Theoretical probabilities
    theoretical_probs = [(1 - 1 / M) ** i * (1 / M) for i in range(n)]
    p_not_found = 1 - sum(theoretical_probs)
    theoretical_probs = [p_not_found] + theoretical_probs

    # Store data for analysis
    results[n] = {
        "indices": list(range(-1, n)),
        "empirical": empirical_freqs,
        "theoretical": theoretical_probs,
    }

    # Plot the graph
    plt.figure(figsize=(10, 5))
    x_positions = np.arange(-1, n)
    plt.bar(
        x_positions - 0.2, theoretical_probs, width=0.4, label="Theoretical", alpha=0.7
    )
    plt.bar(x_positions + 0.2, empirical_freqs, width=0.4, label="Empirical", alpha=0.7)
    plt.yscale("log")  # Log scale for better visualization of small values
    plt.xlabel("Index of Found Key (or -1 = Not Found)")
    plt.ylabel("Probability (log scale)")
    plt.title(f"Linear Search: Empirical vs Theoretical (n = {n}, M = {M})")
    plt.xticks(ticks=x_positions)
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.show()
    plt.savefig(f"linsea_n_{n}.png")

# Tabular display of results
for n in n_values:
    df = pd.DataFrame(
        {
            "Index": results[n]["indices"],
            "Theoretical Probability": results[n]["theoretical"],
            "Empirical Frequency": results[n]["empirical"],
        }
    )
    print(f"\n--- Results for n = {n} ---")
    print(df.to_string(index=False))
