import heapq


import heapq

import osmnx as ox
import networkx as nx
class Algorithms:
    def __init__(self):
        pass

    def dijkstra_with_path(nx.graph G, start, goal):
        # Step 1: Initialize distances and predecessor dictionary
        distances = {node: float("inf") for node in graph}
        predecessors = {node: None for node in graph}  # Track the path
        distances[start] = 0
        # Step 2: Priority Queue (Min-Heap)
        pq = [(0, start)]  # (distance, node)
        while pq:
            current_distance, current_node = heapq.heappop(pq)
            # Step 3: If we reach the goal, stop early
            if current_node == goal:
                break
            # Step 4: Process each neighbor
            for neighbor, weight in graph[current_node]:
                new_distance = current_distance + weight

            # Step 5: If a shorter path is found, update distance & path
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node  # Log predecessor
                heapq.heappush(pq, (new_distance, neighbor))
            # print(predecessors)
            # Step 6: Reconstruct the path from goal to start
        path = []
        current = goal
        while current is not None:
            print(current)
            path.append(current)
            current = predecessors[current]
            path.reverse()  # Reverse to get start -> goal
        return distances, path


# graph = {
#     "A": [("B", 1), ("C", 4)],
#     "B": [("E", 7), ("D", 2)],
#     "C": [("D", 4)],
#     "D": [("F", 6)],
#     "E": [("F", 3)],
#     "F": [],
# }

# # Example Heuristic Values for A*

# start_node = "A"
# goal_node = "F"


# shortest_distance, path = dijkstra_with_path(graph, start_node, goal_node)
# print(f"Shortest Distance: {shortest_distance[goal_node]}")
# print(f"Shortest Path: {path}")
