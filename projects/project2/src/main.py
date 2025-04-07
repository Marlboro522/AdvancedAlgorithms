from fifo_cache import FIFOCache
from belady_cache import BeladyCache
from random_array_gen import generate_array, generate_array_grouped


def simulate_cache(cache, sequence):
    hits = 0
    for page in sequence:
        if cache.get(page):
            hits += 1
    return hits / len(sequence)


def main():
    # Configuration
    capacity = 8
    associativity = 2
    sequence_length = 100000
    alphabet = list(range(101))  # Pages 0–9

    # Generate semirandom sequence
    sequence = generate_array(alphabet, length=sequence_length, repeat_prob=0.6)
    # Uncomment below to test the grouped variant
    sequence = generate_array_grouped(
        alphabet, length=sequence_length, group_size=2, group_repeat_prob=0.75
    )

    print("Generated Sequence of length", len(sequence))
    # print(sequence)

    # FIFO
    fifo_cache = FIFOCache(capacity=capacity, associativity=associativity)
    fifo_hit_rate = simulate_cache(fifo_cache, sequence)
    print(f"FIFO Hit Rate: {fifo_hit_rate:.2f}")

    # Belady
    belady_cache = BeladyCache(
        capacity=capacity, access_sequence=sequence, associativity=associativity
    )
    belady_hit_rate = simulate_cache(belady_cache, sequence)
    print(f"Belady Hit Rate: {belady_hit_rate:.2f}")


if __name__ == "__main__":
    main()
