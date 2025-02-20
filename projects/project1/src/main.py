import osmnx as ox
import networkx as nx
import heapq
import time

# Load Colorado Springs road network
graph_source = "resources/colorado_springs.graphml"
G = ox.load_graphml(graph_source)

# Define source and target nodes
source = 7153623736
target = 555759091

### **1️⃣ Run NetworkX's Built-in Dijkstra**
start_time = time.time()
nx_path = nx.shortest_path(G, source, target, weight="length", method="dijkstra")
nx_distance = nx.shortest_path_length(G, source, target, weight="length")
nx_time = time.time() - start_time

print("\n🔹 NetworkX Dijkstra Results:")
print(f"✔ Shortest Distance: {nx_distance:.2f} meters")
print(f"✔ Number of Nodes in Path: {len(nx_path)}")
print(f"✔ Computation Time: {nx_time:.5f} seconds")


### **2️⃣ Run Our Custom Dijkstra Algorithm**
def dijkstra_osmnx(G, source, target):
    """
    Implements Dijkstra's algorithm for an OSMnx graph.

    Parameters:
        G (networkx.Graph): OSMnx road network graph.
        source (int): Source node ID.
        target (int): Target node ID.

    Returns:
        tuple: (shortest distance, shortest path as list of nodes)
    """
    # Step 1: Initialize distances and priority queue
    distances = {node: float("inf") for node in G.nodes}
    distances[source] = 0
    priority_queue = [(0, source)]
    predecessors = {}

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Stop early if we reach the target
        if current_node == target:
            break

        # Skip processing if we already have a better distance
        if current_distance > distances[current_node]:
            continue

        # Step 2: Explore neighbors
        for neighbor in G.neighbors(current_node):
            edge_data = G.get_edge_data(current_node, neighbor, default={})
            edge_length = edge_data[0].get("length", float("inf"))  # Get road distance

            new_distance = distances[current_node] + edge_length

            # Step 3: Update distances if a shorter path is found
            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_node
                heapq.heappush(priority_queue, (new_distance, neighbor))

    # Step 4: Reconstruct the shortest path
    path = []
    node = target
    while node in predecessors:
        path.append(node)
        node = predecessors[node]
    path.append(source)
    path.reverse()

    return distances[target], path


# Run custom Dijkstra's algorithm
start_time = time.time()
custom_distance, custom_path = dijkstra_osmnx(G, source, target)
custom_time = time.time() - start_time

print("\n🔹 Custom Dijkstra Results:")
print(f"✔ Shortest Distance: {custom_distance:.2f} meters")
print(f"✔ Number of Nodes in Path: {len(custom_path)}")
print(f"✔ Computation Time: {custom_time:.5f} seconds")


### **3️⃣ Verify Results**
print("\n🔹 🔍 Verifying Results:")

# Check if distances match
if abs(nx_distance - custom_distance) < 1e-5:  # Allow a small floating-point error
    print("✅ Shortest distance matches NetworkX!")
else:
    print("❌ Distance mismatch!")

# Check if paths match
if nx_path == custom_path:
    print("✅ Shortest path matches NetworkX!")
else:
    print("❌ Path mismatch!")

# Compare performance
speedup = nx_time / custom_time if custom_time > 0 else float("inf")
print(f"\n🚀 Speed Comparison:")
print(f"Custom Dijkstra is {speedup:.2f}x faster/slower than NetworkX.")
