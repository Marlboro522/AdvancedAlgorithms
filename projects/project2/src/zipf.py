
import lru, fifo_cache, belady_cache
import numpy as np
import matplotlib.pyplot as plt
import os

def generate_zipf_sequence(length, alpha, max_page):
    raw = np.random.zipf(alpha, length)
    return [min(x, max_page) for x in raw]

def run_algo_zipf(algo: str, cache_size: int, num_inputs: int, num_runs: int, page_space: int) -> float:
    rates_per = []
    for _ in range(num_runs):
        input_array = generate_zipf_sequence(num_inputs, alpha=1.2, max_page=page_space)

        if algo.startswith("LRU"):
            ways = {"LRUFull": cache_size, "LRU2way": 2, "LRU4way": 4, "LRU8way": 8}[algo]
            cache = lru.LRUCache(cache_size, ways)
            rates_per.append(cache.runs(input_array))
        elif algo == "FIFO":
            cache = fifo_cache.FIFOCache(cache_size)
            hits = sum(1 for page in input_array if cache.get(page))
            rates_per.append(hits / len(input_array))
        elif algo == "Beladay":
            cache = belady_cache.BeladyCache(cache_size, input_array)
            hits = sum(1 for page in input_array if cache.get(page))
            rates_per.append(hits / len(input_array))
    return sum(rates_per) / num_runs, rates_per

def plot_results(fifo_rates, belady_rates, LRUFull_rates, LRU8way_rates, LRU4way_rates, LRU2way_rates, name_string):
    os.makedirs("zipf_plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.hist(fifo_rates, bins=30, alpha=0.6, label="FIFO")
    plt.hist(belady_rates, bins=30, alpha=0.6, label="Beladys")
    plt.hist(LRUFull_rates, bins=30, alpha=0.6, label="LRU Full")
    plt.hist(LRU8way_rates, bins=30, alpha=0.6, label="LRU 8 Way")
    plt.hist(LRU4way_rates, bins=30, alpha=0.6, label="LRU 4 Way")
    plt.hist(LRU2way_rates, bins=30, alpha=0.6, label="LRU 2 Way")
    plt.xlabel("Hit Rate")
    plt.ylabel("Frequency")
    plt.title("Zipfian Hit Rate Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"zipf_plots/{name_string}_hit_rate_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.boxplot([fifo_rates, belady_rates, LRUFull_rates, LRU8way_rates, LRU4way_rates, LRU2way_rates], labels=["FIFO", "Beladys", "LRU Full", "LRU 8 Way", "LRU 4 Way", "LRU 2 Way"])
    plt.ylabel("Hit Rate")
    plt.title("Zipfian Hit Rate Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"zipf_plots/{name_string}_hit_rate_boxplot.png")

def run_zipf_experiment():
    size = 64
    num_inputs = 10000
    num_runs = 500  
    page_space = 128
    name_string = "zipf"

    results = {}
    results["LRUFull"], LRUFull_rate = run_algo_zipf("LRUFull", size, num_inputs, num_runs, page_space)
    results["LRU8way"], LRUF8way_rate = run_algo_zipf("LRU8way", size, num_inputs, num_runs, page_space)
    results["LRU4way"], LRUF4way_rate = run_algo_zipf("LRU4way", size, num_inputs, num_runs, page_space)
    results["LRU2way"], LRUF2way_rate = run_algo_zipf("LRU2way", size, num_inputs, num_runs, page_space)
    results["FIFO"], fifo_rate = run_algo_zipf("FIFO", size, num_inputs, num_runs, page_space)
    results["Beladay"], belady_rate = run_algo_zipf("Beladay", size, num_inputs, num_runs, page_space)

    plot_results(fifo_rate, belady_rate, LRUFull_rate, LRUF8way_rate, LRUF4way_rate, LRUF2way_rate, name_string)

if __name__ == "__main__":
    run_zipf_experiment()
