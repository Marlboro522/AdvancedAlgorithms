# import osmnx as ox
# import networkx as nx
# import pandas as pd
# import json
# import random
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# import os


# import pandas as pd
# import os


# def precompute_and_save_nearest_transit_nodes_parquet(
#     G, transit_nodes, filename="nearest_transit_nodes.parquet"
# ):
#     length = len(G.nodes())
#     print(
#         f"[INFO] Precomputing nearest transit nodes, this will take time and only done once per map"
#     )
#     if os.path.exists(filename):
#         print(f"[INFO] Loading nearest transit nodes from {filename}")
#         return pd.read_parquet(filename).set_index("node").to_dict()["nearest_transit"]

#     print("[INFO] Precomputing nearest transit nodes...")

#     nearest_transit = []

#     start_time = time.time()
#     for node in G.nodes():
#         length -= 1
#         min_distance = float("inf")
#         nearest_node = None

#         for t in transit_nodes:
#             try:
#                 distance = nx.shortest_path_length(G, node, t, weight="length")
#                 if distance < min_distance:
#                     min_distance = distance
#                     nearest_node = t
#             except nx.NetworkXNoPath:
#                 continue
#         print(f"[INFO] {length} nodes left took {start_time-time.time()} seconds")
#         nearest_transit.append({"node": node, "nearest_transit": nearest_node})

#     df = pd.DataFrame(nearest_transit)

#     # Save to Parquet
#     df.to_parquet(filename, index=False)
#     print(f"[INFO] Nearest transit nodes saved to {filename}")

#     return df.set_index("node").to_dict()["nearest_transit"]


# def convert_tnr_to_dict(tnr_table):
#     """
#     Converts the TNR table DataFrame to a nested dictionary for fast lookups.
#     """
#     print("[DEBUG] Converting TNR table to dictionary...")

#     tnr_dict = {}
#     for _, row in tnr_table.iterrows():
#         node = row["node"]
#         tnr_dict[node] = row["path"]  # Store "data" as the nested dictionary

#     print(f"[DEBUG] TNR dictionary created with {len(tnr_dict)} transit nodes.")
#     return tnr_dict


# def get_distance(s, t, tnr_dict):
#     # return tnr_table.loc[tnr_table["node"] == s]["data"].values[0][t]
#     return tnr_dict.get(s, {}).get(t, float("inf"))


# def load_final_transit_nodes(output_dir="preprocessing_output"):
#     """
#     Loads the final transit nodes from the preprocessed data.
#     Prints the types of stored node IDs to verify format.
#     """
#     print("[DEBUG] Loading final transit nodes...")

#     with open(f"{output_dir}/final_transit_nodes.json", "r") as f:
#         transit_nodes = json.load(f)  # Keep them as stored

#     print(f"[DEBUG] Loaded {len(transit_nodes)} final transit nodes.")
#     # print(f"[DEBUG] Example Transit Nodes: {list(transit_nodes)[:5]}")
#     # print(
#     # f"[DEBUG] Example Transit Node Type: {type(list(transit_nodes)[0])}"
#     # )  # Check type

#     return transit_nodes  # Keep them as strings


# def load_precomputed_tnr_distances(output_dir="preprocessing_output"):
#     """
#     Loads the precomputed transit node distances from Parquet.
#     Prints the types of stored node IDs to verify format.
#     """
#     print("[DEBUG] Loading precomputed TNR distances...")
#     df = pd.read_parquet(f"{output_dir}/transit_node_distances.parquet")

#     print("[DEBUG] Loaded TNR distances successfully.")
#     print(df.head())
#     print(f"[DEBUG] Example Node Type in TNR Table: {type(df['node'].values[0])}")

#     return df  # Keep stored as they are


# # def tnr_query(source, target, nearest_transits, tnr_dict, G, D_local):
# #     """
# #     Computes the shortest path distance using TNR and compares execution time + accuracy with Dijkstra.
# #     Uses precomputed transit node distances where possible.
# #     """
# #     print(f"[DEBUG] Running TNR query from {source} to {target}...\n")

# #     if source not in G.nodes():
# #         print(f"[ERROR] Source node {source} not in the graph.")
# #         return float("inf")

# #     if target not in G.nodes():
# #         print(f"[ERROR] Target node {target} not in the graph.")
# #         return float("inf")

# #     # **Step 1: Measure Dijkstra Time**
# #     dijkstra_start = time.time()
# #     try:
# #         direct_distance = nx.shortest_path_length(G, source, target, weight="length")
# #     except nx.NetworkXNoPath:
# #         print(f"[ERROR] No direct path exists between {source} and {target}.")
# #         return float("inf")

# #     dijkstra_time = time.time() - dijkstra_start
# #     print(f"[DEBUG] Direct Dijkstra Distance: {direct_distance}")
# #     print(f"[DEBUG] Dijkstra Computation Time: {dijkstra_time:.6f} seconds")

# #     # **Step 2: Use Dijkstra If `d(s, t) ≤ D_local`**
# #     if direct_distance <= D_local:
# #         print(f"[DEBUG] Using Direct Dijkstra Distance: {direct_distance}")
# #         return direct_distance

# #     # **Step 3: Find the Nearest Transit Nodes (`u`, `v`)**
# #     print(f"[DEBUG] Finding nearest transit nodes to {source} and {target}...")

# #     tnr_start = time.time()  # Start timing for TNR

# #     nearest_source = nearest_transits.get(source, None)
# #     nearest_target = nearest_transits.get(target, None)
# #     d_su = nx.shortest_path_length(G, source, nearest_source, weight="length")
# #     d_vt = nx.shortest_path_length(G, nearest_target, target, weight="length")
# #     if nearest_source is None or nearest_target is None:
# #         print(
# #             f"[ERROR] No valid transit nodes found for source {source} or target {target}."
# #         )
# #         return float("inf")

# #     print(f"[DEBUG] Nearest Transit Source Node: {nearest_source} ")
# #     print(f"[DEBUG] Nearest Transit Target Node: {nearest_target}")

# #     print(f"[DEBUG] First Leg: {source} → {nearest_source}")

# #     # **Step 4: Look Up `d(u, v)` From `tnr_table`**
# #     transit_distance = get_distance(nearest_source, nearest_target, tnr_dict)
# #     print(
# #         f"[DEBUG] Transit Distance Found: {transit_distance} between {nearest_source} and {nearest_target}"
# #     )

# #     # **Step 5: Compute Final Distance Using TNR Equation**
# #     total_distance = d_su + transit_distance + d_vt
# #     tnr_time = time.time() - tnr_start  # Stop timing for TNR

# #     # **Step 6: Compute Percentage Error**
# #     percentage_error = (abs(total_distance - direct_distance) / direct_distance) * 100

# #     print(f"[DEBUG] TNR Computation Time: {tnr_time:.6f} seconds")
# #     print(f"[DEBUG] Total Estimated Distance: {total_distance}")
# #     print(f"[DEBUG] Percentage Error: {percentage_error:.2f}%\n")

# #     # **Step 7: Print Performance Comparison**
# #     print("\n📊 **Performance Comparison** 📊")
# #     print(f"Dijkstra Time: {dijkstra_time:.6f} seconds")
# #     print(f"TNR Time: {tnr_time:.6f} seconds")
# #     print(f"Speedup: {dijkstra_time / max(tnr_time, 1e-6):.2f}x (TNR is faster if >1)")
# #     print(f"Accuracy: TNR is {percentage_error:.2f}% different from Dijkstra.")

# #     return total_distance


# def tnr_query(source, target, tnr_dict, G, nearest_transit_nodes, D_local):
#     """
#     Computes shortest path using TNR with precomputed nearest-transit-node lookups stored in Parquet.
#     """
#     print(f"[DEBUG] Running TNR query from {source} to {target}...\n")

#     if source not in G.nodes() or target not in G.nodes():
#         print(f"[ERROR] Source or Target not in the graph.")
#         return float("inf")

#     # **Step 1: Measure Dijkstra Time**
#     dijkstra_start = time.time()
#     try:
#         direct_distance = nx.shortest_path_length(G, source, target, weight="length")
#     except nx.NetworkXNoPath:
#         print(f"[ERROR] No direct path exists between {source} and {target}.")
#         return float("inf")
#     dijkstra_time = time.time() - dijkstra_start

#     if direct_distance <= D_local:
#         print(f"[DEBUG] Using Direct Dijkstra Distance: {direct_distance}")
#         return direct_distance

#     # **Step 2: Retrieve Precomputed Nearest Transit Nodes (O(1) Lookup)**
#     nearest_source = nearest_transit_nodes.get(source, None)
#     nearest_target = nearest_transit_nodes.get(target, None)

#     if nearest_source is None or nearest_target is None:
#         print(f"[ERROR] No valid transit nodes found.")
#         return float("inf")

#     print(f"[DEBUG] Nearest Transit Source Node: {nearest_source}")
#     print(f"[DEBUG] Nearest Transit Target Node: {nearest_target}")

#     # **Step 3: Compute d(s, u) and d(v, t)**
#     try:
#         d_su = nx.shortest_path_length(G, source, nearest_source, weight="length")
#         d_vt = nx.shortest_path_length(G, nearest_target, target, weight="length")
#     except nx.NetworkXNoPath:
#         print(f"[ERROR] No valid path to/from transit nodes.")
#         return float("inf")

#     # **Step 4: Look Up d(u, v) From tnr_dict**
#     d_uv = tnr_dict.get(nearest_source, {}).get(nearest_target, float("inf"))

#     if d_uv == float("inf"):
#         print(
#             f"[ERROR] No precomputed distance for {nearest_source} → {nearest_target}"
#         )
#         return float("inf")

#     # **Step 5: Compute Final Distance**
#     total_distance = d_su + d_uv + d_vt  # ✅ Correct formula

#     print(f"[DEBUG] Total Estimated Distance: {total_distance}")
#     return total_distance


# # def compare_one_test(G, transit_nodes, tnr_table):
# #     """
# #     Runs one test comparison between Dijkstra and TNR.
# #     Ensures that source and target exist in G before running queries.
# #     """
# #     # nodes = list(G.nodes())

# #     # Ensure source and target exist in G
# #     while True:
# #         # source, target = random.sample(nodes, 2)
# #         source, target = "506867021", "55859170"
# #         if source in G and target in G:
# #             break  # Found valid nodes
# #         print(f"[DEBUG] Retrying... {source} or {target} is not in G.")
# #     print(type(target))
# #     print(type(source))
# #     print(f"\n[DEBUG] Running test for Source: {source}, Target: {target}")

# #     dist = tnr_query(str(source), str(target), transit_nodes, tnr_table, G)

# #     print(dist)


# def test_tnr_vs_dijkstra(
#     G,
#     nearest_transit_nodes,
#     tnr_dict,
#     D_local_values=[n for n in range(10000, 11000, 1000)],
#     num_tests=10,
# ):
#     """
#     Runs multiple tests comparing TNR vs. Dijkstra with varying `D_local` values and different node distances.
#     Measures speedup and accuracy, then plots the results.
#     """
#     nodes = list(G.nodes())
#     results = []

#     for D_local in D_local_values:
#         print(f"\n🔹 **Testing with D_local = {D_local}** 🔹")

#         for _ in range(num_tests):
#             source, target = random.sample(nodes, 2)

#             print(f"\n[TEST] Source {source} → Target {target} (D_local = {D_local})")

#             # **Step 1: Run Dijkstra**
#             dijkstra_start = time.time()
#             try:
#                 dijkstra_distance = nx.shortest_path_length(
#                     G, source, target, weight="length"
#                 )
#                 dijkstra_valid = True
#             except nx.NetworkXNoPath:
#                 dijkstra_distance = float("inf")
#                 dijkstra_valid = False
#             dijkstra_time = time.time() - dijkstra_start

#             # **Step 2: Run TNR**
#             tnr_start = time.time()
#             tnr_distance = tnr_query(
#                 source,
#                 target,
#                 tnr_dict,
#                 G,
#                 nearest_transit_nodes,
#                 D_local,
#             )
#             tnr_time = time.time() - tnr_start

#             # **Step 3: Compute Speedup & Accuracy**
#             speedup = dijkstra_time / max(tnr_time, 1e-6)
#             percentage_error = (
#                 (abs(tnr_distance - dijkstra_distance) / max(dijkstra_distance, 1))
#                 * 100
#                 if dijkstra_valid
#                 else float("inf")
#             )

#             # **Step 4: Store Results**
#         results.append(
#             {
#                 "D_local": D_local,
#                 "Dijkstra Distance": dijkstra_distance,
#                 "TNR Distance": tnr_distance,
#                 "Dijkstra Time": dijkstra_time,
#                 "TNR Time": tnr_time,
#                 "Speedup": speedup,
#                 "Error (%)": percentage_error,
#             }
#         )

#     # **Step 5: Convert Results to DataFrame & Plot**
#     df = pd.DataFrame(results)

#     # **Plot 1: Speedup vs. D_local**
#     plt.figure(figsize=(8, 5))
#     sns.lineplot(data=df, x="D_local", y="Speedup", marker="o")
#     plt.title("Speedup vs. D_local")
#     plt.xlabel("D_local (meters)")
#     plt.ylabel("Speedup (Dijkstra Time / TNR Time)")
#     plt.grid(True)
#     plt.show()

#     # **Plot 2: Percentage Error vs. D_local**
#     plt.figure(figsize=(8, 5))
#     sns.lineplot(data=df, x="D_local", y="Error (%)", marker="o")
#     plt.title("Accuracy vs. D_local")
#     plt.xlabel("D_local (meters)")
#     plt.ylabel("Percentage Error (%)")
#     plt.grid(True)
#     plt.show()

#     # **Plot 3: TNR vs. Dijkstra Query Time**
#     plt.figure(figsize=(8, 5))
#     sns.scatterplot(data=df, x="Dijkstra Time", y="TNR Time", hue="D_local", s=100)
#     plt.title("TNR vs. Dijkstra Query Time")
#     plt.xlabel("Dijkstra Query Time (seconds)")
#     plt.ylabel("TNR Query Time (seconds)")
#     plt.grid(True)
#     plt.show()

#     return df


# if __name__ == "__main__":
#     """
#     Main execution script:
#     - Loads graph
#     - Loads transit nodes (as integers)
#     - Loads TNR distances (as integers)
#     - Runs one test comparison
#     """
#     print("[DEBUG] Loading graph...")
#     G = nx.read_graphml("resources/colorado_springs.graphml")

#     # Fix edge weight issue (convert to float)
#     print("[DEBUG] Ensuring edge weights are numerical...")
#     for u, v, data in G.edges(data=True):
#         if "length" in data:
#             data["length"] = float(data["length"])  # Convert to float

#     print("[DEBUG] Loading transit nodes and TNR distance data...")
#     transit_nodes = load_final_transit_nodes()
#     tnr_dict = convert_tnr_to_dict(load_precomputed_tnr_distances())

#     nearest_transit_nodes = precompute_and_save_nearest_transit_nodes_parquet(
#         G, transit_nodes
#     )

#     results_df = test_tnr_vs_dijkstra(G, nearest_transit_nodes, tnr_dict)
#     print(results_df)
# # print first five elementns of tnr_table
# # print(tnr_table.head())
# # print first element of tnr_table for node "506867020"

# # print("[DEBUG] Running a single test comparison...")
# # compare_one_test(G, transit_nodes, tnr_table)
# # tnr_table.loc[tnr_table["node"] == "506867020", "data"].values[0]
# # what is the shape of tnr_table
# # print(tnr_table.shape)

import os
import time
import json
import random
import pandas as pd
import networkx as nx
import osmnx as ox
import seaborn as sns
import matplotlib.pyplot as plt


def load_final_transit_nodes(filename="preprocessing_output/final_transit_nodes.json"):
    """
    Loads the final selected transit nodes from a JSON file.

    Args:
        filename (str): Path to the transit nodes JSON file.

    Returns:
        set: Set of selected transit nodes.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"[ERROR] File not found: {filename}")

    with open(filename, "r") as f:
        return set(json.load(f))


def load_precomputed_tnr_distances(
    filename="preprocessing_output/transit_node_distances.parquet",
):
    """
    Loads the precomputed transit node distances from a Parquet file.

    Args:
        filename (str): Path to the Parquet file.

    Returns:
        dict: Dictionary storing shortest path distances between transit nodes.
    """
    print("[DEBUG] Loading precomputed TNR distances...")
    df = pd.read_parquet(filename)
    print("[DEBUG] Loaded TNR distances successfully.")

    tnr_dict = {row["node"]: row["distances"] for _, row in df.iterrows()}
    print(f"[DEBUG] TNR dictionary created with {len(tnr_dict)} transit nodes.")

    return tnr_dict


def load_nearest_transit_nodes(
    filename="preprocessing_output/nearest_transit_nodes.parquet",
):
    """
    Loads the nearest transit nodes for each node from a Parquet file.

    Args:
        filename (str): Path to the Parquet file.

    Returns:
        dict: Dictionary mapping nodes to their nearest transit nodes.
    """
    print(f"[DEBUG] Loading nearest transit nodes from {filename}")
    df = pd.read_parquet(filename)
    return df.set_index("node").to_dict()["nearest_transit"]


def tnr_query(source, target, G, nearest_transit_nodes, tnr_dict, D_local):
    """
    Computes the shortest path using TNR with precomputed nearest transit node lookups.

    Args:
        source (int): Source node ID.
        target (int): Target node ID.
        G (networkx.Graph): The road network graph.
        nearest_transit_nodes (dict): Precomputed nearest transit nodes.
        tnr_dict (dict): Precomputed transit node distances.
        D_local (int): Distance threshold for direct Dijkstra use.

    Returns:
        float: Estimated shortest path distance using TNR.
    """
    print(f"[DEBUG] Running TNR query from {source} to {target}...\n")

    if source not in G.nodes() or target not in G.nodes():
        print(f"[ERROR] Source or Target not in the graph.")
        return float("inf")

    # **Step 1: Measure Dijkstra Time**
    dijkstra_start = time.time()
    try:
        direct_distance = nx.shortest_path_length(G, source, target, weight="length")
    except nx.NetworkXNoPath:
        print(f"[ERROR] No direct path exists between {source} and {target}.")
        return float("inf")

    dijkstra_time = time.time() - dijkstra_start
    print(f"[DEBUG] Direct Dijkstra Distance: {direct_distance}")
    print(f"[DEBUG] Dijkstra Computation Time: {dijkstra_time:.6f} seconds")

    if direct_distance <= D_local:
        print(f"[DEBUG] Using Direct Dijkstra Distance: {direct_distance}")
        return direct_distance

    # **Step 2: Retrieve Precomputed Nearest Transit Nodes (O(1) Lookup)**
    nearest_source = nearest_transit_nodes.get(source, None)
    nearest_target = nearest_transit_nodes.get(target, None)

    if nearest_source is None or nearest_target is None:
        print(f"[ERROR] No valid transit nodes found.")
        return float("inf")

    print(f"[DEBUG] Nearest Transit Source Node: {nearest_source}")
    print(f"[DEBUG] Nearest Transit Target Node: {nearest_target}")

    # **Step 3: Compute d(s, u) and d(v, t)**
    try:
        d_su = nx.shortest_path_length(G, source, nearest_source, weight="length")
        d_vt = nx.shortest_path_length(G, nearest_target, target, weight="length")
    except nx.NetworkXNoPath:
        print(f"[ERROR] No valid path to/from transit nodes.")
        return float("inf")

    # **Step 4: Look Up d(u, v) From tnr_dict**
    d_uv = tnr_dict.get(nearest_source, {}).get(nearest_target, float("inf"))

    if d_uv == float("inf"):
        print(
            f"[ERROR] No precomputed distance for {nearest_source} → {nearest_target}"
        )
        return float("inf")

    # **Step 5: Compute Final Distance**
    total_distance = d_su + d_uv + d_vt  # ✅ Correct formula

    print(f"[DEBUG] Total Estimated Distance: {total_distance}")
    return total_distance


def test_tnr_vs_dijkstra(
    G,
    nearest_transit_nodes,
    tnr_dict,
    D_local_values=[n for n in range(10000, 11000, 1000)],
    num_tests=10,
):
    """
    Runs multiple tests comparing TNR vs. Dijkstra with varying `D_local` values.

    Args:
        G (networkx.Graph): The road network graph.
        nearest_transit_nodes (dict): Precomputed nearest transit nodes.
        tnr_dict (dict): Precomputed transit node distances.
        D_local_values (list, optional): Range of D_local values to test. Defaults to 10,000-11,000.
        num_tests (int, optional): Number of test cases per D_local value. Defaults to 10.

    Returns:
        pandas.DataFrame: Results of the comparisons.
    """
    nodes = list(G.nodes())
    results = []

    for D_local in D_local_values:
        print(f"\n🔹 **Testing with D_local = {D_local}** 🔹")

        for _ in range(num_tests):
            source, target = random.sample(nodes, 2)

            print(f"\n[TEST] Source {source} → Target {target} (D_local = {D_local})")

            # **Step 1: Run Dijkstra**
            dijkstra_start = time.time()
            try:
                dijkstra_distance = nx.shortest_path_length(
                    G, source, target, weight="length"
                )
                dijkstra_valid = True
            except nx.NetworkXNoPath:
                dijkstra_distance = float("inf")
                dijkstra_valid = False
            dijkstra_time = time.time() - dijkstra_start

            # **Step 2: Run TNR**
            tnr_start = time.time()
            tnr_distance = tnr_query(
                source, target, G, nearest_transit_nodes, tnr_dict, D_local
            )
            tnr_time = time.time() - tnr_start

            # **Step 3: Compute Speedup & Accuracy**
            speedup = dijkstra_time / max(tnr_time, 1e-6)
            percentage_error = (
                (abs(tnr_distance - dijkstra_distance) / max(dijkstra_distance, 1))
                * 100
                if dijkstra_valid
                else float("inf")
            )

            # **Step 4: Store Results**
            results.append(
                {
                    "D_local": D_local,
                    "Dijkstra Distance": dijkstra_distance,
                    "TNR Distance": tnr_distance,
                    "Dijkstra Time": dijkstra_time,
                    "TNR Time": tnr_time,
                    "Speedup": speedup,
                    "Error (%)": percentage_error,
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    """
    Main execution script for TNR vs. Dijkstra:
    - Loads precomputed Parquet files
    - Runs comparisons
    """
    print("[DEBUG] Loading graph...")
    G = nx.read_graphml("resources/colorado_springs.graphml")

    print("[DEBUG] Loading precomputed data...")
    nearest_transit_nodes = load_nearest_transit_nodes()
    tnr_dict = load_precomputed_tnr_distances()

    print("[DEBUG] Running tests...")
    results_df = test_tnr_vs_dijkstra(G, nearest_transit_nodes, tnr_dict)

    print(results_df)
