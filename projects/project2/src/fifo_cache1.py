from typing import Optional

class FIFOCache:
    def __init__(self, capacity: int, associativity: int = 0) -> None:
        if capacity <= 0 or (capacity & (capacity - 1)) != 0:
            raise ValueError("Capacity must be a power of 2.")
        if associativity <= 0 or (associativity & (associativity - 1)) != 0:
            raise ValueError("Associativity must be a power of 2.")

        self.capacity = capacity
        self.assoc = associativity
        if self.assoc == 0 or self.assoc > self.capacity:
            self.assoc = self.capacity
        self.sets = capacity // self.assoc
        self.cache = [[-1 for _ in range(self.assoc)] for _ in range(self.sets)]
        self.pointers = [0 for _ in range(self.sets)]

    def get(self, page: int) -> Optional[bool]:
        idx = page % self.sets
        if page in self.cache[idx]:
            return True
        self.put(page)
        return False

    def put(self, page: int) -> None:
        idx = page % self.sets
        self.cache[idx][self.pointers[idx]] = page
        self.pointers[idx] = (self.pointers[idx] + 1) % self.assoc

    def __repr__(self) -> str:
        repr_str = "FIFO Cache Representation:\n"
        repr_str += "-" * 50 + "\n"
        repr_str += "Cache Contents:\n"
        repr_str += f"{'Set':<5} | {'Way Contents':<20}\n"
        repr_str += "-" * 50 + "\n"
        for set_idx, ways in enumerate(self.cache):
            repr_str += f"{set_idx:<5} | {', '.join(map(str, ways)):<20}\n"
        repr_str += "-" * 50 + "\n"
        return repr_str

def test_fifo_cache():
    print("Running FIFO Cache Tests")
    cache = FIFOCache(capacity=4, associativity=2)
    assert not cache.get(1), "Test 1 Failed: MISS expected"
    assert cache.get(1), "Test 2 Failed: HIT expected"
    assert not cache.get(2), "Test 3 Failed: MISS expected"
    assert not cache.get(3), "Test 4 Failed: MISS expected"
    assert not cache.get(4), "Test 5 Failed: MISS expected"
    assert cache.get(3), "Test 6 Failed: HIT expected"
    assert not cache.get(5), "Test 7 Failed: MISS expected"
    print("FIFO Cache Tests Passed")

if __name__ == "__main__":
    test_fifo_cache()
