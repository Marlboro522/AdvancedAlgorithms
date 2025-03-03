import os
import json
import time
import multiprocessing
import osmnx as ox
import geopandas as gpd
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import box


def load_graph(path: str):
    """
    Loads a road network graph from a GraphML file.

    Args:
        path (str): Path to the GraphML file.

    Returns:
        tuple: NetworkX graph, GeoDataFrames of nodes and edges.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    G = ox.load_graphml(path)
    nodes, edges = ox.convert.graph_to_gdfs(G, nodes=True, edges=True)
    return G, nodes, edges


def find_bounding_box(G):
    north = max(G.nodes[node]["y"] for node in G.nodes)
    south = min(G.nodes[node]["y"] for node in G.nodes)
    east = max(G.nodes[node]["x"] for node in G.nodes)
    west = min(G.nodes[node]["x"] for node in G.nodes)
    return north, south, east, west


def create_square_grid(north, south, east, west, output_geojson, grid_size=10):
    if os.path.exists(output_geojson):
        print(f"File already exists: {output_geojson}")
        return
    lat_range = north - south
    lon_range = east - west
    step_size = min(lat_range, lon_range) / grid_size
    grid_cells = []

    for i in range(grid_size):
        for j in range(grid_size):
            cell_west = west + j * step_size
            cell_east = cell_west + step_size
            cell_south = south + i * step_size
            cell_north = cell_south + step_size
            grid_cells.append(box(cell_west, cell_south, cell_east, cell_north))

    grid_gdf = gpd.GeoDataFrame(geometry=grid_cells, crs="EPSG:4326")
    grid_gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"Grid saved to {output_geojson}")
    return grid_gdf


def find_nearest_transit_node(G, node, transit_nodes):
    """
    Finds the nearest transit node for a given node.

    Args:
        G (networkx.Graph): The road network graph.
        node (int): The node for which we are finding the nearest transit.
        transit_nodes (set): Set of transit nodes.

    Returns:
        tuple: (node, nearest_transit_node)
    """
    global remaining_nodes  # Use a global variable for tracking remaining nodes

    min_distance = float("inf")
    nearest_node = None

    for t in transit_nodes:
        try:
            distance = nx.shortest_path_length(G, node, t, weight="length")
            if distance < min_distance:
                min_distance = distance
                nearest_node = t
        except nx.NetworkXNoPath:
            continue

    # Decrement the counter safely
    with remaining_nodes.get_lock():
        remaining_nodes.value -= 1
        total_nodes = len(G.nodes())  # Get total number of nodes
        processed_nodes = total_nodes - remaining_nodes.value  # Nodes that are done

        if remaining_nodes.value < 0:
            print(f"[WARNING] Remaining nodes went negative! Resetting to 0.")
            remaining_nodes.value = 0

        if remaining_nodes.value % 100 == 0:  # Print every 100 nodes
            print(f"[INFO] {remaining_nodes.value} nodes remaining...")
            print(f"[DEBUG] Processed {processed_nodes}/{total_nodes} nodes.")

    return node, nearest_node


def precompute_nearest_transit_nodes(
    G, transit_nodes, filename="preprocessing_output/nearest_transit_nodes.parquet"
):
    """
    Computes the nearest transit node for every node in the graph using multiprocessing.

    Args:
        G (networkx.Graph): The road network graph.
        transit_nodes (set): Set of transit nodes.
        filename (str): Path to save the Parquet file.

    Returns:
        dict: Dictionary mapping nodes to their nearest transit nodes.
    """
    if os.path.exists(filename):
        print(f"[INFO] Loading nearest transit nodes from {filename}")
        return pd.read_parquet(filename).set_index("node").to_dict()["nearest_transit"]

    print("[INFO] Precomputing nearest transit nodes using multiprocessing...")
    print(f"[INFO] Transit nodes selected: {len(transit_nodes)}")

    start_time = time.time()
    num_workers = min(multiprocessing.cpu_count(), 8)  # Use up to 8 processes

    print(f"[INFO] Using {num_workers} parallel processes")

    # Initialize a shared counter inside the worker pool
    global remaining_nodes
    remaining_nodes = multiprocessing.Value("i", len(G.nodes()))

    with multiprocessing.Pool(processes=num_workers, initializer=init_worker) as pool:
        results = pool.starmap(
            find_nearest_transit_node,
            [(G, node, transit_nodes) for node in G.nodes()],
        )

    # Convert results to DataFrame
    df = pd.DataFrame(results, columns=["node", "nearest_transit"])

    # Save to Parquet
    df.to_parquet(filename, index=False)
    print(f"[INFO] Nearest transit nodes saved to {filename}")

    print(f"[INFO] Computation Time: {(time.time() - start_time) / 60:.2f} minutes")

    return df.set_index("node").to_dict()["nearest_transit"]


def init_worker():
    """
    Initializes worker processes with a shared counter.
    """
    global remaining_nodes
    remaining_nodes = multiprocessing.Value("i", 0)  # Initialize in each worker


def compute_transit_node_distances(
    G, transit_nodes, filename="preprocessing_output/transit_node_distances.parquet"
):
    """
    Computes shortest paths between all transit nodes and saves them as a Parquet file.

    Args:
        G (networkx.Graph): The road network graph.
        transit_nodes (set): Set of transit nodes.
        filename (str): Path to save the Parquet file.

    Returns:
        dict: Dictionary storing shortest paths between transit nodes.
    """
    if os.path.exists(filename):
        print(f"[INFO] Loading precomputed transit node distances from {filename}")
        return pd.read_parquet(filename).set_index("node").to_dict()["distances"]

    print("[INFO] Computing transit node distances...")

    tnr_data = []
    for node in transit_nodes:
        shortest_paths = nx.single_source_dijkstra(G, node, weight="length")[0]
        filtered_paths = {
            t: shortest_paths[t] for t in transit_nodes if t in shortest_paths
        }
        tnr_data.append({"node": node, "distances": filtered_paths})

    df = pd.DataFrame(tnr_data)

    # **DEBUG: Check DataFrame before saving**
    print(f"[DEBUG] Transit Node Distances DataFrame:\n{df.head()}")

    df.to_parquet(filename, index=False)
    print(f"[INFO] Transit node distances saved to {filename}")

    return df.set_index("node").to_dict()["distances"]


def assign_nodes_to_grid_and_find_boundaries(
    G, nodes, grid_gdf, output_dir="preprocessing_output"
):
    """
    Assigns nodes to grid cells and finds edges that cross boundaries.

    Args:
        G (networkx.Graph): The road network graph.
        nodes (geopandas.GeoDataFrame): The nodes GeoDataFrame.
        grid_gdf (geopandas.GeoDataFrame): The grid GeoDataFrame.
        output_dir (str, optional): Path to save output JSON files.

    Returns:
        tuple: (set of transit nodes, boundary crossing edges).
    """
    if os.path.exists(f"{output_dir}/boundary_crossing_edges.json"):
        print(f"[INFO] Boundary crossing edges already exist. Skipping computation.")
        return load_final_transit_nodes(output_dir), load_boundary_crossing_edges(
            output_dir
        )

    os.makedirs(output_dir, exist_ok=True)

    node_to_grid = {}
    for node in G.nodes():
        node_lat, node_lon = nodes.loc[node, "y"], nodes.loc[node, "x"]
        for i, cell in grid_gdf.iterrows():
            if cell.geometry.contains(gpd.points_from_xy([node_lon], [node_lat])[0]):
                node_to_grid[node] = i
                break

    boundary_crossing_edges = [
        (u, v) for u, v in G.edges() if node_to_grid.get(u) != node_to_grid.get(v)
    ]

    transit_nodes = {node for edge in boundary_crossing_edges for node in edge}

    with open(f"{output_dir}/boundary_crossing_edges.json", "w") as f:
        json.dump(boundary_crossing_edges, f)
    with open(f"{output_dir}/final_transit_nodes.json", "w") as f:
        json.dump(list(transit_nodes), f)

    return transit_nodes, boundary_crossing_edges


def compute_betweenness_for_transit_nodes(
    G, transit_nodes, filename="preprocessing_output/transit_nodes_centrality.parquet"
):
    """
    Computes betweenness centrality for the full graph but stores only transit nodes' values.

    Args:
        G (networkx.Graph): The full road network graph.
        transit_nodes (set): Nodes identified as crossing grid boundaries.
        filename (str): Path to save the betweenness centrality results.

    Returns:
        dict: Betweenness centrality values for transit nodes.
    """
    if os.path.exists(filename):
        print(f"[INFO] Loading precomputed betweenness centrality from {filename}")
        return pd.read_parquet(filename).set_index("node").to_dict()["betweenness"]

    print("[INFO] Computing betweenness centrality for the full graph...")

    centrality = nx.betweenness_centrality(G, weight="length")
    filtered_centrality = {
        node: centrality[node] for node in transit_nodes if node in centrality
    }

    df = pd.DataFrame(
        list(filtered_centrality.items()), columns=["node", "betweenness"]
    )
    df.to_parquet(filename, index=False)

    print(f"[INFO] Transit node betweenness centrality saved to {filename}")
    return filtered_centrality


def select_top_transit_nodes(
    centrality_dict,
    percentile=95,
    filename="preprocessing_output/final_transit_nodes.json",
):
    """
    Selects the top X% of transit nodes based on betweenness centrality.

    Args:
        centrality_dict (dict): Betweenness centrality values for transit nodes.
        percentile (int): Percentile threshold.
        filename (str): Path to save selected transit nodes.

    Returns:
        set: Set of selected transit nodes.
    """
    values = list(centrality_dict.values())
    threshold = np.percentile(values, percentile)
    final_transit_nodes = {
        node for node, centrality in centrality_dict.items() if centrality >= threshold
    }

    with open(filename, "w") as f:
        json.dump(list(final_transit_nodes), f)

    print(
        f"[INFO] Selected {len(final_transit_nodes)} transit nodes (Top {percentile}%)"
    )
    return final_transit_nodes


def preprocessing_pipeline(G, nodes, grid_gdf):
    """
    Runs the full preprocessing pipeline and returns all necessary Parquet and JSON files.

    Args:
        G (networkx.Graph): The road network graph.
        nodes (geopandas.GeoDataFrame): The nodes GeoDataFrame.
        grid_gdf (geopandas.GeoDataFrame): The grid GeoDataFrame.

    Returns:
        dict: Dictionary containing paths to all generated files.
    """
    transit_nodes, boundary_crossing_edges = assign_nodes_to_grid_and_find_boundaries(
        G, nodes, grid_gdf
    )
    centrality_dict = compute_betweenness_for_transit_nodes(G, transit_nodes)
    final_transit_nodes = select_top_transit_nodes(centrality_dict, percentile=95)

    nearest_transit_nodes = precompute_nearest_transit_nodes(G, final_transit_nodes)
    transit_node_distances = compute_transit_node_distances(G, final_transit_nodes)

    # Return all file paths
    return {
        "boundary_crossing_edges": "preprocessing_output/boundary_crossing_edges.json",
        "final_transit_nodes": "preprocessing_output/final_transit_nodes.json",
        "transit_nodes_centrality": "preprocessing_output/transit_nodes_centrality.parquet",
        "nearest_transit_nodes": "preprocessing_output/nearest_transit_nodes.parquet",
        "transit_node_distances": "preprocessing_output/transit_node_distances.parquet",
    }


def load_final_transit_nodes(
    output_dir="preprocessing_output/final_transit_nodes.json",
):
    """
    Loads the final transit nodes from a JSON file.

    Args:
        output_dir (str): Path to the transit nodes JSON file.

    Returns:
        set: Set of selected transit nodes.
    """
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"[ERROR] File not found: {output_dir}")

    with open(output_dir, "r") as f:
        return set(json.load(f))


def load_boundary_crossing_edges(
    output_dir="preprocessing_output/boundary_crossing_edges.json",
):
    """
    Loads the boundary crossing edges from a JSON file.

    Args:
        output_dir (str): Path to the boundary crossing edges JSON file.

    Returns:
        list: List of boundary crossing edges.
    """
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"[ERROR] File not found: {output_dir}")

    with open(output_dir, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    """
    Main execution script for preprocessing:
    - Loads graph
    - Runs preprocessing pipeline
    - Saves all required JSON & Parquet files
    """
    start = time.time()
    print("[DEBUG] Loading graph...")
    G, nodes, edges = load_graph("resources/colorado_springs.graphml")

    grid_gdf = create_square_grid(*find_bounding_box(G), "resources/grid.geojson")

    # ✅ Run preprocessing & store generated file paths
    json_files = preprocessing_pipeline(G, nodes, grid_gdf)

    print("[INFO] Preprocessing completed successfully!")
    print("[INFO] Generated Files:")
    for key, path in json_files.items():
        print(f"  {key}: {path}")
    print(
        "[INFO] Preprocessing took {:.2f} minutes.".format((time.time() - start) / 60)
    )
