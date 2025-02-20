import osmnx as os
import networkx as nx
import random

colorado_springs = "resources/colorado_springs.graphml"

G = os.load_graphml(colorado_springs)

print(f"Nodes: {len(G.nodes)} Edges: {len(G.edges)}")

node = random.choice(list(G.nodes))

lat, lon = G.nodes[node]["y"], G.nodes[node]["x"]
print(f"Node: {node} Latitude: {lat} Longitude: {lon}")
# bounding box
north = max(G.nodes[node]["y"] for node in G.nodes)
south = min(G.nodes[node]["y"] for node in G.nodes)
east = max(G.nodes[node]["x"] for node in G.nodes)
west = min(G.nodes[node]["x"] for node in G.nodes)

# get the nodes within the bounding box
print(f"North: {north}, South: {south}, East: {east}, West: {west}")
