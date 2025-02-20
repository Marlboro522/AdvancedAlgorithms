import osmnx as ox
import networkx as nx
import random

from shapely import length

colorado_springs = "resources/colorado_springs.graphml"

G = ox.load_graphml(colorado_springs)

print(f"Nodes: {len(G.nodes)} Edges: {len(G.edges)}")

# node = random.choice(list(G.nodes))


# lat, lon = G.nodes[node]["y"], G.nodes[node]["x"]
# print(f"Node: {node} Latitude: {lat} Longitude: {lon}")
# # bounding box
# north = max(G.nodes[node]["y"] for node in G.nodes)
# south = min(G.nodes[node]["y"] for node in G.nodes)
# east = max(G.nodes[node]["x"] for node in G.nodes)
# west = min(G.nodes[node]["x"] for node in G.nodes)

# # get the nodes within the bounding box
# print(f"North: {north}, South: {south}, East: {east}, West: {west}")

# node = 12352350091

# for neighbor in G.neighbors(node):
#     print(f"Neighboring Node: {neighbor}")
#     edge_data = G.get_edge_data(node, neighbor, default={})
#     if len(edge_data) == 0:
#         print(f"No edge between {node} and {neighbor}")
#     else:
#         for key, data in edge_data.items():  # Iterate over all edges (if multiple)
#             print(
#                 f"{key}th edge id: {data.get('osmid', 'Unknown')} length is {data.get('length', 'Unknown'):.2f} meters"
#             )
#     for key, data in edge_data.items():
#         print(f"{key} : {data}")
#     print("\n")
# Select a node
# node = 12352350091
# lat1, lon1 = G.nodes[node]["y"], G.nodes[node]["x"]
# print(f"Node: {node} Latitude: {lat1} Longitude: {lon1}")

# # Select a neighbor
# neighbor_node = random.choice(list(G.neighbors(node)))
# lat2, lon2 = G.nodes[neighbor_node]["y"], G.nodes[neighbor_node]["x"]

# print(f"Neighbor Node: {neighbor_node} Latitude: {lat2} Longitude: {lon2}")

# # Compute Great-Circle Distance (in meters)
# great_circle_distance = ox.distance.great_circle(lat1, lon1, lat2, lon2)
# print(f"Great Circle Distance: {great_circle_distance:.2f} meters")


def get_lat_lon(node):
    return G.nodes[node]["y"], G.nodes[node]["x"]


source = list(G.nodes)[0]
target = list(G.nodes)[-1]

print(f"Source: {source} Target: {target}")
shortest_path_distance = nx.shortest_path_length(G, source, target, weight="length")
source_lat, source_lon = get_lat_lon(source)
target_lat, target_lon = get_lat_lon(target)
great_circle_distance = ox.distance.great_circle(
    source_lat, source_lon, target_lat, target_lon
)
print(f"Shortest Path Distance: {shortest_path_distance} meters")
print(f"Great Circle Distance: {great_circle_distance} meters")

# percentage difference
percentage_difference = (
    (shortest_path_distance - great_circle_distance) / great_circle_distance * 100
)
print(f"Percentage Difference: {percentage_difference:.2f}%")
