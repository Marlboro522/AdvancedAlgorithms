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
north = max(G.nodes[node]["y"] for node in G.nodes)
south = min(G.nodes[node]["y"] for node in G.nodes)
east = max(G.nodes[node]["x"] for node in G.nodes)
west = min(G.nodes[node]["x"] for node in G.nodes)

# get the nodes within the bounding box
print(f"North: {north}, South: {south}, East: {east}, West: {west}")

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
# import time

# start_time = time.time()
# centrality = nx.betweenness_centrality(G, weight="length")
# end_time = time.time()

# # start_time = time.time()
# # centrality = nx.betweenness_centrality(G, weight="length")
# # end_time = time.time()

# print(
#     f"Centrality Calculation Time for coclroado Springs is: {(end_time - start_time)/60:.6f} Minutes."
# )
# # print(centrality)

# most_central_node = max(centrality, key=centrality.get)
# most_central_score = centrality[most_central_node]

# print(
#     f"Most Central Node: {most_central_node} with centrality score: {most_central_score:.5f}"
# )

# Nodes: 20662 Edges: 50842
# Centrality Calculation Time for coclroado Springs is: 90.789877 Minutes.
# Most Central Node: 55744303 with centrality score: 0.12405

import pandas as pd
import networkx as nx
import random
import json


def load_final_transit_nodes(output_dir="preprocessing_output"):
    """
    Loads the final transit nodes from the preprocessed data.
    Uses `final_transit_nodes.json` instead of the wrong Parquet file.
    """
    print("[DEBUG] Loading final transit nodes...")

    with open(f"{output_dir}/final_transit_nodes.json", "r") as f:
        transit_nodes = set(map(str, json.load(f)))  # Ensure IDs are strings

    print(f"[DEBUG] Loaded {len(transit_nodes)} final transit nodes.")
    print(f"[DEBUG] Example Transit Nodes: {list(transit_nodes)[:5]}")
    return transit_nodes


def load_precomputed_tnr_distances(output_dir="preprocessing_output"):
    print("[DEBUG] Loading precomputed TNR distances...")
    df = pd.read_parquet(f"{output_dir}/transit_node_distances.parquet")
    print("[DEBUG] Loaded TNR distances successfully.")
    print(df.head())
    return df


def tnr_query(source, target, transit_nodes, tnr_table, G):
    """
    Computes the shortest path distance using TNR, utilizing the precomputed transit node distances.
    """
    print(f"[DEBUG] Running TNR query from {source} to {target}...")

    if str(source) not in G or str(target) not in G:
        print(f"[ERROR] One of the nodes ({source}, {target}) is not in the graph.")
        return float("inf")

    # Convert all node IDs to strings for consistency
    transit_nodes = set(map(str, transit_nodes))
    G_nodes = set(map(str, G.nodes()))
    valid_transit_nodes = {t for t in transit_nodes if t in G_nodes}

    if not valid_transit_nodes:
        print(f"[ERROR] No valid transit nodes found in the graph.")
        return float("inf")

    try:
        # **STEP 1: Find the nearest transit nodes to the source and target**
        nearest_source = min(
            valid_transit_nodes,
            key=lambda t: nx.shortest_path_length(G, str(source), t, weight="length"),
        )
        nearest_target = min(
            valid_transit_nodes,
            key=lambda t: nx.shortest_path_length(G, str(target), t, weight="length"),
        )

        print(f"[DEBUG] Nearest Transit Nodes: {nearest_source} → {nearest_target}")

        # **STEP 2: Extract the Precomputed Distance Correctly**
        if nearest_source in tnr_table["node"].values:
            row = tnr_table.loc[tnr_table["node"] == nearest_source, "data"].values[0]
            print(row)
            if nearest_target in row:
                transit_distance = row[nearest_target]
                print(
                    f"[DEBUG] Transit Distance Found: {transit_distance} between {nearest_source} and {nearest_target}"
                )
            else:
                print(
                    f"[ERROR] No precomputed distance for {nearest_source} → {nearest_target}"
                )
                return float("inf")
        else:
            print(f"[ERROR] No transit node data for {nearest_source}")
            return float("inf")

        # **STEP 3: Compute the full shortest path**
        return (
            nx.shortest_path_length(G, str(source), nearest_source, weight="length")
            + transit_distance
            + nx.shortest_path_length(G, nearest_target, str(target), weight="length")
        )

    except nx.NetworkXNoPath:
        print(
            f"[ERROR] No transit node reachable from source {source} or target {target}."
        )
        return float("inf")


def dijkstra_query(G, source, target):
    """
    Computes the shortest path distance using Dijkstra’s Algorithm.
    Returns -1 if the path does not exist.
    """
    print(f"[DEBUG] Running Dijkstra query from {source} to {target}...")

    if source not in G or target not in G:
        print(f"[ERROR] One of the nodes ({source}, {target}) is not in the graph.")
        return float("inf")

    try:
        return nx.shortest_path_length(G, source, target, weight="length")
    except nx.NetworkXNoPath:
        print(f"[ERROR] No path exists between {source} and {target}.")
        return float("inf")


def compare_one_test(G, transit_nodes, tnr_table):
    nodes = list(G.nodes())
    source, target = random.sample(nodes, 2)

    print(f"\n[DEBUG] Running test for Source: {source}, Target: {target}")

    dijkstra_dist = dijkstra_query(G, source, target)
    tnr_dist = tnr_query(source, target, transit_nodes, tnr_table, G)

    error = abs(dijkstra_dist - tnr_dist) / max(
        dijkstra_dist, 1
    )  # Avoid division by zero

    print("\n[RESULTS]")
    print(f"Dijkstra Distance = {dijkstra_dist}")
    print(f"TNR Distance = {tnr_dist}")
    print(f"Error Percentage = {error:.2%}")


if __name__ == "__main__":
    print("[DEBUG] Loading graph...")
    G = nx.read_graphml("resources/colorado_springs.graphml")

    # Fix edge weight issue (convert to float)
    print("[DEBUG] Ensuring edge weights are numerical...")
    for u, v, data in G.edges(data=True):
        if "length" in data:
            data["length"] = float(data["length"])  # Convert to float

    print("[DEBUG] Loading transit nodes and TNR distance data...")
    transit_nodes = load_final_transit_nodes()
    tnr_table = load_precomputed_tnr_distances()

    print("[DEBUG] Running a single test comparison...")
    compare_one_test(G, transit_nodes, tnr_table)
