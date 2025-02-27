# import osmnx as ox
# import networkx as nx
# import pandas as pd
# import json
# import random
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns


# def convert_tnr_to_dict(tnr_table):
#     """
#     Converts the TNR table DataFrame to a nested dictionary for fast lookups.
#     """
#     print("[DEBUG] Converting TNR table to dictionary...")

#     tnr_dict = {}
#     for _, row in tnr_table.iterrows():
#         node = row["node"]
#         tnr_dict[node] = row["data"]  # Store "data" as the nested dictionary

#     print(f"[DEBUG] TNR dictionary created with {len(tnr_dict)} transit nodes.")
#     return tnr_dict


# def get_distance(s, t, tnr_dict):
#     # return tnr_table.loc[tnr_table["node"] == s]["data"].values[0][t]
#     tnr_dict.get(s, {}).get(t, float("inf"))


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
#     # print(df.head())
#     # print(f"[DEBUG] Example Node Type in TNR Table: {type(df['node'].values[0])}")

#     return df  # Keep stored as they are


# def tnr_query(source, target, transit_nodes, tnr_table, G, D_local):
#     """
#     Computes the shortest path distance using TNR and compares execution time + accuracy with Dijkstra.
#     Uses precomputed transit node distances where possible.
#     """
#     print(f"[DEBUG] Running TNR query from {source} to {target}...\n")

#     if source not in G.nodes():
#         print(f"[ERROR] Source node {source} not in the graph.")
#         return float("inf")

#     if target not in G.nodes():
#         print(f"[ERROR] Target node {target} not in the graph.")
#         return float("inf")

#     # **Step 1: Measure Dijkstra Time**
#     dijkstra_start = time.time()
#     try:
#         direct_distance = nx.shortest_path_length(G, source, target, weight="length")
#     except nx.NetworkXNoPath:
#         print(f"[ERROR] No direct path exists between {source} and {target}.")
#         return float("inf")

#     dijkstra_time = time.time() - dijkstra_start
#     print(f"[DEBUG] Direct Dijkstra Distance: {direct_distance}")
#     print(f"[DEBUG] Dijkstra Computation Time: {dijkstra_time:.6f} seconds")

#     # **Step 2: Use Dijkstra If `d(s, t) ≤ D_local`**
#     if direct_distance <= D_local:
#         print(f"[DEBUG] Using Direct Dijkstra Distance: {direct_distance}")
#         return direct_distance

#     # **Step 3: Find the Nearest Transit Nodes (`u`, `v`)**
#     print(f"[DEBUG] Finding nearest transit nodes to {source} and {target}...")

#     tnr_start = time.time()  # Start timing for TNR

#     nearest_source, min_source_dist = None, float("inf")
#     for t in transit_nodes:
#         try:
#             distance = nx.shortest_path_length(G, source, t, weight="length")
#             if distance < min_source_dist:
#                 min_source_dist = distance
#                 nearest_source = t
#         except nx.NetworkXNoPath:
#             continue

#     nearest_target, min_target_dist = None, float("inf")
#     for t in transit_nodes:
#         try:
#             distance = nx.shortest_path_length(G, target, t, weight="length")
#             if distance < min_target_dist:
#                 min_target_dist = distance
#                 nearest_target = t
#         except nx.NetworkXNoPath:
#             continue

#     if nearest_source is None or nearest_target is None:
#         print(
#             f"[ERROR] No valid transit nodes found for source {source} or target {target}."
#         )
#         return float("inf")

#     print(
#         f"[DEBUG] Nearest Transit Source Node: {nearest_source} (Distance: {min_source_dist})"
#     )
#     print(
#         f"[DEBUG] Nearest Transit Target Node: {nearest_target} (Distance: {min_target_dist})"
#     )

#     print(f"[DEBUG] First Leg: {source} → {nearest_source}")

#     # **Step 4: Look Up `d(u, v)` From `tnr_table`**
#     transit_distance = get_distance(nearest_source, nearest_target, tnr_table)
#     print(
#         f"[DEBUG] Transit Distance Found: {transit_distance} between {nearest_source} and {nearest_target}"
#     )

#     # **Step 5: Compute Final Distance Using TNR Equation**
#     total_distance = min_source_dist + transit_distance + min_target_dist
#     tnr_time = time.time() - tnr_start  # Stop timing for TNR

#     # **Step 6: Compute Percentage Error**
#     percentage_error = (abs(total_distance - direct_distance) / direct_distance) * 100

#     print(f"[DEBUG] TNR Computation Time: {tnr_time:.6f} seconds")
#     print(f"[DEBUG] Total Estimated Distance: {total_distance}")
#     print(f"[DEBUG] Percentage Error: {percentage_error:.2f}%\n")

#     # **Step 7: Print Performance Comparison**
#     print("\n📊 **Performance Comparison** 📊")
#     print(f"Dijkstra Time: {dijkstra_time:.6f} seconds")
#     print(f"TNR Time: {tnr_time:.6f} seconds")
#     print(f"Speedup: {dijkstra_time / max(tnr_time, 1e-6):.2f}x (TNR is faster if >1)")
#     print(f"Accuracy: TNR is {percentage_error:.2f}% different from Dijkstra.")

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

# import random
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns


# def test_tnr_vs_dijkstra(
#     G,
#     transit_nodes,
#     tnr_table,
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
#                 source, target, transit_nodes, tnr_table, G, D_local
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
#     tnr_table = load_precomputed_tnr_distances()
#     print(tnr_table.head())
#     tnr_dict = convert_tnr_to_dict(tnr_table)
#     test= tnr_dict.get('55771189',{}).get('10005025643',float('inf'))
#     print(test)

#     # results_df = test_tnr_vs_dijkstra(G, transit_nodes, tnr_table)
#     # print(results_df.head())
# # print first five elementns of tnr_table
# # print(tnr_table.head())
# # print first element of tnr_table for node "506867020"

# # print("[DEBUG] Running a single test comparison...")
# # compare_one_test(G, transit_nodes, tnr_table)
# # tnr_table.loc[tnr_table["node"] == "506867020", "data"].values[0]
# # what is the shape of tnr_table
# # print(tnr_table.shape)


import osmnx as ox
import time
import networkx as nx

G = ox.load_graphml("resources/colorado_springs.graphml")
print(f"Nodes Count = {len(G.nodes())} and Edge Count = {len(G.edges())}")
star_time = time.time()
cetrality = nx.betweenness_centrality(G, weight="length")
print(f"Time taken = {time.time()-star_time}")
