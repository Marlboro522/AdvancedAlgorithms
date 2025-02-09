# # from collections import deque

# # # Add elements to the deque.
# # queue = deque([2,3,1,2,3,5])
# # print(queue)


# # # Remove and return the leftmost element.
# # print(queue.popleft())
# # print(queue)


# # # Remove and return the rightmost element.
# # print(queue.pop())
# # print(queue)


# # # Add elements to the right.
# # queue.extend([4,5])
# # print(queue)


# # # Add elements to the left using the appendleft() method.
# # queue.appendleft(0)
# # print(queue)

# #prioroty queue implementaitonn using deque
# from collections import deque
# import heapq

# # Create a min heap
# heap = []
# heapq.heappush(heap, 10)
# heapq.heappush(heap, 5)
# heapq.heappush(heap, 8)
# heapq.heappush(heap, 3)

# print(heap)
# # Pop and print the smallest element
# print(heapq.heappop(heap))
# print(heap)

# # Push a new element
# heapq.heappush(heap, 7)
# print(heap)

# # Pop and print the smallest element
# print(heapq.heappop(heap))
# print(heap)

# # Push a new element, but only if it is greater than the smallest element in the heap
# heapq.heappush(heap, 2)
# print(heap)

# # Pop and print the smallest element
# print(heapq.heappop(heap))
# print(heap)

# import heapq


# def dijkstra_with_paths(graph, start, goal=None):
#     # Initialize distances and predecessor dictionary
#     distances = {node: float("inf") for node in graph}
#     predecessors = {node: None for node in graph}
#     distances[start] = 0

#     # Priority queue: (distance, node)
#     pq = []
#     heapq.heappush(pq, (0, start))

#     while pq:
#         current_distance, current_node = heapq.heappop(pq)

#         # If goal is specified and reached, stop early
#         if goal and current_node == goal:
#             break

#         # Explore neighbors
#         for neighbor, weight in graph[current_node]:
#             distance = current_distance + weight
#             if distance < distances[neighbor]:
#                 distances[neighbor] = distance
#                 predecessors[neighbor] = current_node
#                 heapq.heappush(pq, (distance, neighbor))

#     return distances, predecessors


# def reconstruct_path(predecessors, start, goal):
#     path = []
#     current = goal
#     while current is not None:
#         path.append(current)
#         current = predecessors[current]
#     path.reverse()  # Reverse to get start -> goal
#     return path if path[0] == start else None  # Check if a valid path exists


# # Function to handle user input for the graph
# def create_graph():
#     print("Enter the graph as adjacency list:")
#     print("Example: {'A': [('B', 1), ('C', 4)], 'B': [('C', 2), ('D', 5)], ...}")
#     graph_input = input("Graph: ")
#     graph = eval(graph_input)
#     return graph

# graph = {
#     "A": [("B", 1), ("C", 4)],
#     "B": [("E", 7), ("D", 2)],
#     "C": [("D", 4)],
#     "D": [("F", 6)],
#     "E": [("F", 3)],
#     "F": [],
# }

# start_node = "A"
# goal_node = "F"

# distances, predecessors = dijkstra_with_paths(graph, start_node, goal_node)
# path = reconstruct_path(predecessors, start_node, goal_node)

# print(f"Shortest path from {start_node} to {goal_node}: {path}")
# print(f"Distances: {distances}")





import heapq


pq=[]
heapq.heappush(pq,"A")
heapq.heappush(pq,"B")
heapq.heappush(pq,"C")

print(heapq.heappop(pq))
print(pq)