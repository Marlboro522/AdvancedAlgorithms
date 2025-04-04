from typing import Optional, List

class BeladyCache:
    def __init__(self, capacity: int, access_sequence: List[int], associativity: int = 0) -> None:
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
        self.sequence = access_sequence
        self.current_index = 0

    def get(self, page: int) -> Optional[bool]:
        idx = page % self.sets
        if page in self.cache[idx]:
            self.current_index += 1
            return True
        self.put(page)
        self.current_index += 1
        return False

    def put(self, page: int) -> None:
        idx = page % self.sets
        set_cache = self.cache[idx]
        if -1 in set_cache:
            insert_pos = set_cache.index(-1)
            self.cache[idx][insert_pos] = page
        else:
            future = self.sequence[self.current_index+1:]
            distance = []
            for cached_page in set_cache:
                if cached_page in future:
                    distance.append(future.index(cached_page))
                else:
                    distance.append(float('inf'))
            victim = distance.index(max(distance))
            self.cache[idx][victim] = page

    def __repr__(self) -> str:
        repr_str = "Belady's OPT Cache Representation:\n"
        repr_str += "-" * 50 + "\n"
        repr_str += "Cache Contents:\n"
        repr_str += f"{'Set':<5} | {'Way Contents':<20}\n"
        repr_str += "-" * 50 + "\n"
        for set_idx, ways in enumerate(self.cache):
            repr_str += f"{set_idx:<5} | {', '.join(map(str, ways)):<20}\n"
        repr_str += "-" * 50 + "\n"
        return repr_str

# -------------------------------
# ✅ Test Cases
# -------------------------------
def test_belady_cache():
    print("Running Belady Cache Tests")
    sequence = [1, 2, 3, 1, 2, 4, 5, 6, 1, 2, 3]
    cache = BeladyCache(capacity=4, access_sequence=sequence, associativity=2)
    
    assert not cache.get(1), "Test 1 Failed: MISS expected"
    assert not cache.get(2), "Test 2 Failed: MISS expected"
    assert not cache.get(3), "Test 3 Failed: MISS expected"
    assert cache.get(1), "Test 4 Failed: HIT expected"
    assert cache.get(2), "Test 5 Failed: HIT expected"
    assert not cache.get(4), "Test 6 Failed: MISS expected"
    assert not cache.get(5), "Test 7 Failed: MISS expected"
    assert not cache.get(6), "Test 8 Failed: MISS expected"
    assert cache.get(1), "Test 9 Failed: HIT expected"
    
    print("✅ Belady Cache Tests Passed")

if __name__ == "__main__":
    test_belady_cache()
