import heapq


def dijkstra(graph, start_node):
    # Step 1: Initialize distances and priority queue
    distances = {node: float("inf") for node in graph}  # d[v] = ∞ for all nodes v
    distances[start_node] = 0  # d[start_node] = 0

    # PriorityQueue Q = {all nodes in Graph}
    pq = [(0, start_node)]  # Stores (distance, node)

    while pq:  # While Q is not empty
        # Step 2: Extract u = node in Q with smallest d[u]
        current_distance, u = heapq.heappop(pq)
        print(graph[u])

        # Step 3: For each neighbor v of u
        for neighbor, weight in graph[u]:
            print(f"Neighbor: {neighbor}, Weight: {weight}")
            # Relaxation: Update distances if a shorter path is found
            if current_distance + weight < distances[neighbor]:
                distances[neighbor] = current_distance + weight
                # Add the updated distance to the priority queue
                heapq.heappush(pq, (distances[neighbor], neighbor))

    return distances


import heapq


def a_star(graph, start_node, goal_node, heuristic):
    # Step 1: Initialize distances and priority queue
    distances = {node: float("inf") for node in graph}  # d[v] = ∞ for all nodes v
    distances[start_node] = 0  # d[start_node] = 0

    # Priority queue stores (distance + heuristic, distance, node)
    pq = []
    heapq.heappush(pq, (0 + heuristic[start_node], 0, start_node))  # (f(u), g(u), node)

    while pq:  # While Q is not empty
        # Log the priority queue
        print(f"Priority Queue: {pq}")

        # Step 2: Extract u = node in Q with smallest d[u] + h(u)
        f_u, current_distance, u = heapq.heappop(pq)

        # Step 3: If u == goal_node, return the shortest distance
        if u == goal_node:
            return distances[goal_node]

        # Step 4: For each neighbor v of u
        for neighbor, weight in graph[u]:
            tentative_distance = current_distance + weight  # g(u) + weight(u, v)

            # Relaxation step: Update distances if a shorter path is found
            if tentative_distance < distances[neighbor]:
                distances[neighbor] = tentative_distance
                f_v = tentative_distance + heuristic[neighbor]  # f(v) = g(v) + h(v)
                heapq.heappush(
                    pq, (f_v, tentative_distance, neighbor)
                )  # Push (f(v), g(v), v)

    return distances[goal_node] if distances[goal_node] < float("inf") else None


# Example Graph as an adjacency list
graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("E", 7), ("D", 2)],
    "C": [("D", 4)],
    "D": [("F", 6)],
    "E": [("F", 3)],
    "F": [],
}

# Example Heuristic Values for A*
heuristic = {
    "A": 10,
    "B": 8,
    "C": 6,
    "D": 4,
    "E": 2,
    "F": 0,
}

start_node = "A"
goal_node = "F"

shortest_distance = a_star(graph, start_node, goal_node, heuristic)
if shortest_distance is not None:
    print(f"Shortest distance from {start_node} to {goal_node}: {shortest_distance}")
else:
    print(f"No path exists from {start_node} to {goal_node}")


# graph = {
#     "A": [("B", 1), ("C", 4)],
#     "B": [("E", 7), ("D", 2)],
#     "C": [("D", 4)],
#     "D": [("F", 6)],
#     "E": [("F", 3)],
#     "F": [],
# }

# start_node = "A"
# # goal_node = "F"

# # print(graph.keys())
# shortest_distance = dijkstra(graph, start_node)
# print(f"Shortest distance from {start_node}: {shortest_distance}")
