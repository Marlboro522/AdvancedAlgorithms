from math import log2
from typing import Any, Dict, Optional
import random


class LRUCache:
    def __init__(self, capacity: int, associativity: int = 0) -> None:
        """
        Initialize the LRU Cache with a given capacity.
        :param capacity: Maximum number of items the cache can hold.
        """

        self.capacity: int = capacity  # maximum number of items the cache can hold
        self.assoc: int = associativity  # associativity of the cache
        if self.assoc == 0 or self.assoc > self.capacity:
            self.assoc = self.capacity
        # Check if capacity is a power of 2
        if self.capacity <= 0 or (self.capacity & (self.capacity - 1)) != 0:
            raise ValueError("Capacity must be a power of 2.")

        # Check if associativity is a power of 2
        if self.assoc <= 0 or (self.assoc & (self.assoc - 1)) != 0:
            raise ValueError("Associativity must be a power of 2.")

        self.LRUbits: int = int(log2(self.assoc))
        self.LRUlevels: list[list[int]] = []  # List of arrays for each LRU bit
        self.sets: int = capacity // self.assoc

        # Initialize LRUlevels with arrays of increasing size (powers of 2)
        for bit in range(self.LRUbits):
            self.LRUlevels.append([0] * (capacity // (2 ** (bit + 1))))

        # Initialize self.cache as a list of size self.sets, with each inner list being size self.assoc
        self.cache: list[list[int]] = [
            [-1 for _ in range(self.assoc)] for _ in range(self.sets)
        ]
        # outer index is set index, inner index is way

    def get(self, page: int) -> Optional[Any]:
        """
        Retrieve an item from the cache.
        :param page: The key to look up in the cache.
        :return: True on hit, false on miss
        """
        # find set index
        set_idx = page % self.sets
        for way, value in enumerate(self.cache[set_idx]):
            if page == value:  # hit
                # update lru bits since we are using it
                for bit, level in enumerate(self.LRUlevels):
                    div = 2 ** (bit + 1)
                    entry = set_idx * self.assoc + way
                    level[entry // div] = (entry + 1) % 2
                return True  # and return we got a hit
        # miss
        self.put(page)
        return False

    def put(self, page: int) -> None:
        """
        Add an item to the cache. If the cache exceeds its capacity, evict the least recently used item.
        :param page: The key to add to the cache.
        """
        # find set index
        idx = page % self.sets
        # traverse the LRU levels to find the least recently used item, flipping bits along the way
        for level in reversed(self.LRUlevels):
            bit = level[idx]
            level[idx] = (level[idx] + 1) % 2
            idx = idx * 2 + bit
        # evict the least recently used item
        self.cache[idx // self.assoc][idx % self.assoc] = page

    def runs(self, inputArray) -> float:
        hit_count: int = 0
        for page in inputArray:
            hit = self.get(page)
            if hit:
                hit_count += 1
        return hit_count / len(inputArray)

    def __repr__(self) -> str:
        """
        Return a string representation of the cache for debugging purposes.
        This includes the LRU levels and the cache contents in columns.
        """
        repr_str = "LRU Cache Representation:\n"
        repr_str += "-" * 50 + "\n"

        # Add LRU levels
        repr_str += "LRU Levels:\n"
        for i, level in enumerate(reversed(self.LRUlevels)):
            repr_str += f"Level {i}: {level}\n"

        repr_str += "-" * 50 + "\n"

        # Add cache contents
        repr_str += "Cache Contents:\n"
        repr_str += f"{'Set':<5} | {'Way Contents':<20}\n"
        repr_str += "-" * 50 + "\n"
        for set_idx, ways in enumerate(self.cache):
            repr_str += f"{set_idx:<5} | {', '.join(map(str, ways)):<20}\n"

        repr_str += "-" * 50 + "\n"
        return repr_str


def test_lru_cache():
    cache_size = 8
    associativity = 4
    cache = LRUCache(cache_size, associativity)
    inputs = [i for i in range(1, 17)]
    # Expected cache states after each insertion
    expected_cache = [
        [[-1, -1, -1, -1], [1, -1, -1, -1]],  # After inserting 1
        [[2, -1, -1, -1], [1, -1, -1, -1]],  # After inserting 2
        [[2, -1, -1, -1], [1, -1, 3, -1]],  # After inserting 3
        [[2, -1, 4, -1], [1, -1, 3, -1]],  # After inserting 4
        [[2, -1, 4, -1], [1, 5, 3, -1]],  # After inserting 5
        [[2, 6, 4, -1], [1, 5, 3, -1]],  # After inserting 6
        [[2, 6, 4, -1], [1, 5, 3, 7]],  # After inserting 7
        [[2, 6, 4, 8], [1, 5, 3, 7]],  # After inserting 8
        [[2, 6, 4, 8], [9, 5, 3, 7]],  # After inserting 9
        [[10, 6, 4, 8], [9, 5, 3, 7]],  # After inserting 10
        [[10, 6, 4, 8], [9, 5, 11, 7]],  # After inserting 11
        [[10, 6, 12, 8], [9, 5, 11, 7]],  # After inserting 12
        [[10, 6, 12, 8], [9, 13, 11, 7]],  # After inserting 13
        [[10, 14, 12, 8], [9, 13, 11, 7]],  # After inserting 14
        [[10, 14, 12, 8], [9, 13, 11, 15]],  # After inserting 15
        [[10, 14, 12, 16], [9, 13, 11, 15]],  # After inserting 16
    ]

    # Expected LRU levels after each insertion
    expected_lru_levels = [
        [[0, 0, 1, 0], [0, 1]],  # After inserting 1
        [[1, 0, 1, 0], [1, 1]],  # After inserting 2
        [[1, 0, 1, 1], [1, 0]],  # After inserting 3
        [[1, 1, 1, 1], [0, 0]],  # After inserting 4
        [[1, 1, 0, 1], [0, 1]],  # After inserting 5
        [[0, 1, 0, 1], [1, 1]],  # After inserting 6
        [[0, 1, 0, 0], [1, 0]],  # After inserting 7
        [[0, 0, 0, 0], [0, 0]],  # After inserting 8
        [[0, 0, 1, 0], [0, 1]],  # After inserting 9
        [[1, 0, 1, 0], [1, 1]],  # After inserting 10
        [[1, 0, 1, 1], [1, 0]],  # After inserting 11
        [[1, 1, 1, 1], [0, 0]],  # After inserting 12
        [[1, 1, 0, 1], [0, 1]],  # After inserting 13
        [[0, 1, 0, 1], [1, 1]],  # After inserting 14
        [[0, 1, 0, 0], [1, 0]],  # After inserting 15
        [[0, 0, 0, 0], [0, 0]],  # After inserting 16
    ]

    # Perform the test
    for i, page in enumerate(inputs):
        print(f"Inserting page {page}...")
        cache.get(page)

        # Check the cache state
        assert cache.cache == expected_cache[i], f"Cache state mismatch at step {i + 1}"
        assert (
            cache.LRUlevels == expected_lru_levels[i]
        ), f"LRU levels mismatch at step {i + 1}"

        # Print the cache state for debugging
        print(cache)

    print("All tests passed!")


def test_any(cache_size, associativity, page_space, num_lookups):
    # Cache configuration
    """cache_size = 8
    associativity = 8
    page_space = 12  # Pages range from 1 - page_space
    num_lookups = 16  # Number of random page lookups"""
    hit_count = 0
    # Initialize the LRUCache
    cache = LRUCache(capacity=cache_size, associativity=associativity)
    # inputs = [x for x in range(1,17)]

    print("Initial Cache State:")
    print(cache)

    # Perform random page lookups
    print("\nPerforming random page lookups...")
    for _ in range(num_lookups):
        page = random.randint(1, page_space)
        hit = cache.get(page)
        if hit:
            hit_count += 1
        print(f"Accessing page {page}: {'HIT' if hit else 'MISS'}")
        # print(cache)
        # input(...)
    print(cache)
    hit_rate = hit_count / num_lookups
    print(f"Hit rate: {hit_rate}")


if __name__ == "__main__":
    # test_any(size,ways, page range, num inputs)

    # test_any(8,8,12,100) #size 8, full assoc(full LRU), 12 page range, 100 lookups
    test_any(32, 8, 64, 1000)  # size 32, 8way, 64 page range, 1000 lookups

    # test_lru_cache()
