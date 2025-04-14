import lru, fifo_cache, belady_cache
from randomArrayGen import generate_array,generate_array_grouped
import typing
import random
import matplotlib.pyplot as plt
"""
do N runs of each algorithm, or algorithm variant
average cache hit rate
dict of {algo: avg_hit rate}
plot algo vs avg_hit rate

"""

def run_algo(algo: str, cache_size: int, inputSource: str, num_inputs:int, num_runs: int, page_space: int) -> float:
    '''
    Instantiates the given algo data structure, and 
    '''
    inputArray = [int]
    alpha = [x for x in range(1,page_space)] # want to get rid of alpha just use range(n1,n2)

    for _ in range(num_runs):
        cache_hit_rate_sum: float = 0
        if inputSource == "true-random":
            inputArray = [random.randint(1,page_space) for _ in range(num_inputs)]
        elif inputSource == "biased-individual":
            inputArray = generate_array(alpha,num_inputs,repeat_prob=0.25)
        elif inputSource == "biased-grouped":
            inputArray = generate_array_grouped(alpha,num_inputs,4,group_repeat_prob=0.25)

        if algo == "LRUFull":
            cache = lru.LRUCache(cache_size, cache_size)
        elif algo == "LRU2way":
            cache = lru.LRUCache(cache_size, 2)
        elif algo == "LRU4way":
            cache = lru.LRUCache(cache_size, 4)
        elif algo == "LRU8way":
            cache = lru.LRUCache(cache_size, 8)
        elif algo == "FIFO":
            cache = fifo_cache.FIFOCache(cache_size)
        elif algo == "Belady":
            cache = belady_cache.BeladyCache(cache_size,inputArray)
        else:
            print(f"{algo} does not match any algorithm")

        cache_hit_rate_sum += cache.runs(inputArray)

    return cache_hit_rate_sum/num_runs
        
def collect_data(inputSource: str, size: int,num_inputs: int, num_runs:int, page_space: int) -> dict[str,float]:
    results:dict[str, float] = {}
    results["LRUFull"] = run_algo("LRUFull", size,inputSource,num_inputs,num_runs, page_space)
    results["LRU8way"] = run_algo("LRU8way", size,inputSource,num_inputs,num_runs, page_space)
    results["LRU4way"] = run_algo("LRU4way", size,inputSource,num_inputs,num_runs, page_space)
    results["LRU2way"] = run_algo("LRU2way", size,inputSource,num_inputs,num_runs, page_space)

    results["FIFO"] =   run_algo("FIFO", size,inputSource,num_inputs, num_runs, page_space)
    results["Belady"] = run_algo("Belady", size,inputSource,num_inputs, num_runs, page_space)

    return results


def plot_data(data: dict,inputSource: str) -> None:
    """
    Plots a bar chart for the average cache hit rates of different algorithms.
    """
    algorithms = list(data.keys())
    hit_rates = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.bar(algorithms, hit_rates, color='skyblue')
    plt.xlabel('Algorithms', fontsize=14)
    plt.ylabel('Average Cache Hit Rate', fontsize=14)
    plt.title(f'Cache Hit Rate Comparison: {inputSource}', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_data_comparison(data_sets: dict[str, dict[str, float]]) -> None:
    """
    Plots a line chart comparing the average cache hit rates of different algorithms
    across multiple input sources.
    """
    plt.figure(figsize=(10, 6))

    for input_source, data in data_sets.items():
        algorithms = list(data.keys())
        hit_rates = list(data.values())
        plt.plot(algorithms, hit_rates, marker='o', label=input_source)

    plt.xlabel('Algorithms', fontsize=14)
    plt.ylabel('Average Cache Hit Rate', fontsize=14)
    plt.title('Cache Hit Rate Comparison Across Input Sources', fontsize=16)
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title="Input Sources", fontsize=12)
    plt.tight_layout()
    plt.show()   

if __name__=="__main__":
    data1 = collect_data("true-random", 64, 10000, 10, 128)
    plot_data(data1,"true-random")

    data2 = collect_data("biased-individual", 64, 10000, 10, 128)
    plot_data(data2,"biased-individual")

    data3 = collect_data("biased-grouped", 64, 10000, 10, 128)
    plot_data(data3,"biased-grouped")

        # Combine data sets for comparison
    combined_data = {
        "true-random": data1,
        "biased-individual": data2,
        "biased-grouped": data3,
    }
    plot_data_comparison(combined_data)