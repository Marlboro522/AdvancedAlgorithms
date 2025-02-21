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


# source = list(G.nodes)[0]
# target = list(G.nodes)[-1]

# print(f"Source: {source} Target: {target}")
# shortest_path_distance = nx.shortest_path_length(G, source, target, weight="length")
# source_lat, source_lon = get_lat_lon(source)
# target_lat, target_lon = get_lat_lon(target)
# great_circle_distance = ox.distance.great_circle(
#     source_lat, source_lon, target_lat, target_lon
# )
# print(f"Shortest Path Distance: {shortest_path_distance} meters")
# print(f"Great Circle Distance: {great_circle_distance} meters")

# # percentage difference
# percentage_difference = (
#     (shortest_path_distance - great_circle_distance) / great_circle_distance * 100
# )
# print(f"Percentage Difference: {percentage_difference:.2f}%")


# def find_longest_edge(G):
#     longest_edge_length = 0

#     for u, v in G.edges():
#         edge_data = G.get_edge_data(u, v, default={})

#         if edge_data:
#             longest_edge_between_nodes = 0

#             for _, data in edge_data.items():
#                 longest_edge_between_nodes = max(
#                     longest_edge_between_nodes, data.get("length", 0)
#                 )

#             longest_edge_length = max(longest_edge_length, longest_edge_between_nodes)

#     return longest_edge_length


# edge_length = find_longest_edge(G)
# print(f"Longest Edge Length: {edge_length:.2f} meters")


# convnerting the mapo innto a geodataframe

# import geopandas as gpd

# gdf_nodes, gdf_edges = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)

# print(gdf_nodes.head(5))
# print(gdf_edges.head(5))

# total_road_length = gdf_edges["length"].sum()

# print(f"Total edge length according to osmnx is {ox.stats.edge_length_total(G)}")
# print(
#     f"Total Road Length in Colorado Springs: {total_road_length:.2f} meters and {total_road_length/1000:.2f} km"
# )

# road_type_counts = gdf_edges["highway"].value_counts()

# print(road_type_counts)

# most_common_road = road_type_counts.idxmax()
# most_common_count = road_type_counts.max()


# print(f"Most Common Road Type: {most_common_road} with {most_common_count} roads")

centrality = nx.betweenness_centrality(G, weight="length")

print(centrality)

most_central_node = max(centrality, key=centrality.get)
most_central_score = centrality[most_central_node]

print(
    f"Most Central Node: {most_central_node} with centrality score: {most_central_score:.5f}"
)

most_central_node = max(centrality, key=centrality.get)
most_central_score = centrality[most_central_node]

print(
    f"Most Central Node: {most_central_node} with centrality score: {most_central_score:.5f}"
)
