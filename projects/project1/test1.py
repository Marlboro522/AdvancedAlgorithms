import osmnx as ox
import networkx as nx

import matplotlib.pyplot as plt

graph = ox.load_graphml("colorado_springs.graphml")

print(f"Number of nodes: {graph.number_of_nodes()}")
print(f"Number of edges: {graph.number_of_edges()}")

print(f"Density: {nx.density(graph)}")
