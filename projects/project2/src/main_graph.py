from fifo_cache import FIFOCache
from belady_cache import BeladyCache
from random_array_gen import generate_array

import sys

print(sys.executable)

import matplotlib.pyplot as plt
import numpy as np
import sys

print(sys.executable)
def simulate_cache(cache, sequence):
    hits = 0
    for page in sequence:
        if cache.get(page):
            hits += 1
    return hits / len(sequence)

def run_experiments(trials=1000, capacity=8, sequence_length=100, repeat_prob=0.6):
    fifo_rates = []
    belady_rates = []
    alphabet = list(range(10))

    for _ in range(trials):
        sequence = generate_array(alphabet, length=sequence_length, repeat_prob=repeat_prob)

        fifo_cache = FIFOCache(capacity=capacity)
        fifo_hit_rate = simulate_cache(fifo_cache, sequence)
        fifo_rates.append(fifo_hit_rate)

        belady_cache = BeladyCache(capacity=capacity, access_sequence=sequence)
        belady_hit_rate = simulate_cache(belady_cache, sequence)
        belady_rates.append(belady_hit_rate)

    return fifo_rates, belady_rates

def plot_results(fifo_rates, belady_rates):
    plt.figure(figsize=(10, 5))

    # Histogram
    plt.hist(fifo_rates, bins=30, alpha=0.6, label='FIFO')
    plt.hist(belady_rates, bins=30, alpha=0.6, label="Belady's OPT")
    plt.xlabel("Hit Rate")
    plt.ylabel("Frequency")
    plt.title("Hit Rate Distribution over Multiple Trials")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hit_rate_distribution.png")
    plt.show()

    # Box plot
    plt.figure(figsize=(8, 5))
    plt.boxplot([fifo_rates, belady_rates], labels=["FIFO", "Belady's OPT"])
    plt.ylabel("Hit Rate")
    plt.title("Hit Rate Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("hit_rate_boxplot.png")
    plt.show()

def main():
    fifo_rates, belady_rates = run_experiments()
    print(f"FIFO Average Hit Rate: {np.mean(fifo_rates):.4f}")
    print(f"Belady Average Hit Rate: {np.mean(belady_rates):.4f}")
    plot_results(fifo_rates, belady_rates)

if __name__ == "__main__":
    main()
