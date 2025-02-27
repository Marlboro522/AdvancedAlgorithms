import osmnx as ox
import geopandas as gpd
import os
import json
import networkx as nx
from shapely.geometry import box
import matplotlib.pyplot as plt
import multiprocessing


def load_graph(path: str):
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


def plot_grid_with_map(G, grid_gdf, output_file):
    if os.path.exists(output_file):
        print(f"File already exists: {output_file}")
        return
    fig, ax = plt.subplots(figsize=(10, 10))
    ox.plot_graph(
        G,
        ax=ax,
        node_size=0,
        edge_color="gray",
        edge_linewidth=0.5,
        show=False,
        close=False,
    )
    grid_gdf.boundary.plot(ax=ax, edgecolor="red", linewidth=0.5, alpha=0.7)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Grid map saved to {output_file}")


def load_final_transit_nodes(output_dir="preprocessing_output"):
    with open(f"{output_dir}/final_transit_nodes.json", "r") as f:
        transit_nodes = set(map(int, json.load(f)))
    return transit_nodes


def assign_nodes_to_grid_and_find_boundaries(
    G, nodes, grid_gdf, output_dir="preprocessing_output"
):
    if os.path.exists(f"{output_dir}/node_to_grid.json"):
        print(f"File already exists: {output_dir}/node_to_grid.json")
        return
    os.makedirs(output_dir, exist_ok=True)

    node_to_grid = {}

    for node in G.nodes():
        node_lat, node_lon = nodes.loc[node, "y"], nodes.loc[node, "x"]

        for i, cell in grid_gdf.iterrows():
            if cell.geometry.contains(gpd.points_from_xy([node_lon], [node_lat])[0]):
                node_to_grid[node] = i
                break

    print(f"Assigned {len(node_to_grid)} nodes to grid cells")

    with open(f"{output_dir}/node_to_grid.json", "w") as f:
        json.dump(node_to_grid, f)
    print(f"Saved node-to-grid mapping: {output_dir}/node_to_grid.json")

    boundary_crossing_edges = []
    for u, v in G.edges():
        if node_to_grid.get(u) != node_to_grid.get(v):
            boundary_crossing_edges.append((u, v))

    print(
        f"Total edges: {len(G.edges())}, Found {len(boundary_crossing_edges)} boundary-crossing edges"
    )

    with open(f"{output_dir}/boundary_crossing_edges.json", "w") as f:
        json.dump(boundary_crossing_edges, f)
    print(f"Saved boundary-crossing edges: {output_dir}/boundary_crossing_edges.json")

    transit_nodes = set()
    for u, v in boundary_crossing_edges:
        transit_nodes.add(u)
        transit_nodes.add(v)

    print(
        f"Total transit nodes: {len(transit_nodes)}, Total boundary nodes: {len(boundary_crossing_edges)}"
    )

    with open(f"{output_dir}/final_transit_nodes.json", "w") as f:
        json.dump(list(transit_nodes), f)
    print(f"Saved transit nodes: {output_dir}/final_transit_nodes.json")


def filter_final_transit_nodes(G, output_dir):

    transit_nodes = load_final_transit_nodes(output_dir)

    print(f"Loaded {len(transit_nodes)} transit node candidates.")

    boundary_edges_path = f"{output_dir}/boundary_crossing_edges.json"
    with open(boundary_edges_path, "r") as f:
        boundary_crossing_edges = json.load(f)

    node_edge_count = {node: 0 for node in transit_nodes}

    for u, v in boundary_crossing_edges:
        if u in transit_nodes:
            node_edge_count[u] += 1
        if v in transit_nodes:
            node_edge_count[v] += 1

    final_transit_nodes = {}

    for node, edge_count in node_edge_count.items():
        if edge_count < 2:
            continue
        if G.degree(node) < 3:
            continue
        final_transit_nodes[node] = True

    print(f"Final transit nodes selected: {len(final_transit_nodes)}")

    with open(f"{output_dir}/final_transit_nodes.json", "w") as f:
        json.dump(final_transit_nodes, f)

    print(f"Saved final transit nodes: {output_dir}/final_transit_nodes.json")


def compute_shortest_paths(G, node, transit_nodes):
    shortest_paths_length, shortest_paths_nodes = nx.single_source_dijkstra(
        G, node, weight="length"
    )

    return node, {
        "distances": {
            t: shortest_paths_length.get(t, float("inf"))
            for t in transit_nodes
            if t in shortest_paths_length
        },
        "paths": {
            t: shortest_paths_nodes.get(t, [])
            for t in transit_nodes
            if t in shortest_paths_nodes
        },
    }


def compute_transit_node_distances(G, output_dir="preprocessing_output"):
    os.makedirs(output_dir, exist_ok=True)

    transit_nodes = load_final_transit_nodes(output_dir)

    print(
        f"Computing distances and paths for {len(transit_nodes)} transit nodes uruns on all avilablee cores/ "
    )

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    results = pool.starmap(
        compute_shortest_paths, [(G, node, transit_nodes) for node in transit_nodes]
    )
    pool.close()
    pool.join()

    tnr_table = {node: data["distances"] for node, data in results}
    # tnr_paths = {node: data["paths"] for node, data in results}

    with open(f"{output_dir}/transit_node_distances.json", "w") as f:
        json.dump(tnr_table, f)

    # with open(f"{output_dir}/transit_node_paths.json", "w") as f:
    #     json.dump(tnr_paths, f)

    print(f"Saved transit node distances in {output_dir}")


def main():
    output_geojson = "resources/grid.geojson"
    output_img = "resources/grid_map.png"
    output_dir = "preprocessing_output"
    place = "resources/colorado_springs.graphml"

    G, nodes, edges = load_graph(place)

    north, south, east, west = find_bounding_box(G)
    print(f"Nodes: {len(nodes)} Edges: {len(edges)}")

    grid_gdf = create_square_grid(north, south, east, west, output_geojson)
    plot_grid_with_map(G, grid_gdf, output_img)

    assign_nodes_to_grid_and_find_boundaries(G, nodes, grid_gdf, output_dir)
    filter_final_transit_nodes(G, output_dir)
    compute_transit_node_distances(G, output_dir)


import time

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Preprocessing took {(time.time()-start)/60} minutes")
