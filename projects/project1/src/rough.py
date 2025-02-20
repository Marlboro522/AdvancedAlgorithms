import random
import osmnx as ox
import networkx as nx
import time
import matplotlib.pyplot as plt

# Load the Colorado Springs graph
graph_source = "resources/colorado_springs.graphml"
G = ox.load_graphml(graph_source)


s, t = 7153623736, 555759091


start_time = time.time()


shortest_path = nx.shortest_path(
    G, source=s, target=t, weight="length", method="dijkstra"
)

end_time = time.time()
print(f"Dijkstra Computation Time: {end_time - start_time:.5f} seconds")


fig, ax = ox.plot_graph_route(
    G, shortest_path, route_linewidth=4, node_size=30, bgcolor="white"
)


def custom_variant(G, source, target):
    pass
