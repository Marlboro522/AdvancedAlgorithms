import osmnx as ox
import networkx as nx
import os

# place_name = "Colorado Springs, Colorado, USA"

# G = ox.graph_from_place(place_name, network_type="drive", simplify=True)
graph_source = "resources/colorado_springs.graphml"
G = ox.load_graphml(graph_source)

nodes, edges = ox.graph_to_gdfs(G)
path = os.path.join("data", "data.txt")
# number of nnodes in the graph.
print(len(G.nodes))
with open(path, "w") as f:
    f.write(f"++" * 10 + "nodes" + "++" * 10)
    f.write("\n")
    f.write(str(nodes))
    f.write("\n")
    f.write("++" * 10 + "edges" + "++" * 10)
    f.write("\n")
    f.write(str(edges))

print(f"Data written to {path}")
# print(help(ox.graph_to_gdfs(G)))
