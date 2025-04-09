
class BeladyCache:
    def __init__(self, capacity: int, access_sequence: list[int]) -> None:
        self.capacity = capacity
        self.sequence = access_sequence
        self.cache = []
        self.current_index = 0

    def get(self, page: int) -> bool:
        if page in self.cache:
            self.current_index += 1
            return True
        self.put(page)
        self.current_index += 1
        return False

    def put(self, page: int) -> None:
        if len(self.cache) < self.capacity:
            self.cache.append(page)
        else:
            future = self.sequence[self.current_index+1:]
            distances = []
            for cached_page in self.cache:
                if cached_page in future:
                    distances.append(future.index(cached_page))
                else:
                    distances.append(float('inf'))
            victim_index = distances.index(max(distances))
            self.cache[victim_index] = page

    def __repr__(self) -> str:
        return f"Belady Cache: {self.cache}"
