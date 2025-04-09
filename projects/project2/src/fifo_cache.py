
class FIFOCache:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache = []
        self.index = 0

    def get(self, page: int) -> bool:
        if page in self.cache:
            return True
        self.put(page)
        return False

    def put(self, page: int) -> None:
        if len(self.cache) < self.capacity:
            self.cache.append(page)
        else:
            self.cache[self.index] = page
            self.index = (self.index + 1) % self.capacity

    def __repr__(self) -> str:
        return f"FIFO Cache: {self.cache}"
