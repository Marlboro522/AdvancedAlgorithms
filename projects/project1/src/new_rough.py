import osmnx as ox
import networkx as nx
G = ox.load_graphml("resources/colorado_springs.graphml")

source = "55771189"
target = "546459916"

# shortest_path = ox.shortest_path(G, source, target, weight="length")
# print(f"Shortest Path: {shortest_path}")
# shortest_path_length = nx.shortest_path_length(G, source, target, weight="length")
# print(f"Shortest Path Length: {shortest_path_length:.2f} meters")

shortest_path_length_nx = nx.shortest_path_length(G, source, target, weight="length")
print(f"Shortest Path Lengthinn networkx: {shortest_path_length_nx:.2f} meters")
