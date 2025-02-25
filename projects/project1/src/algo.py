import json
import networkx as nx
import random


def load_final_transit_nodes(output_dir="preprocessing_output"):
    print("[DEBUG] Loading final transit nodes...")
    with open(f"{output_dir}/final_transit_nodes.json", "r") as f:
        transit_nodes = set(map(int, json.load(f)))  # Ensure IDs are integers
    print(f"[DEBUG] Loaded {len(transit_nodes)} transit nodes.")
    return transit_nodes


def load_precomputed_tnr_data(output_dir="preprocessing_output"):
    print("[DEBUG] Loading precomputed TNR data...")
    with open(f"{output_dir}/transit_node_distances.json", "r") as f:
        tnr_table = json.load(f)

    with open(f"{output_dir}/transit_node_paths.json", "r") as f:
        tnr_paths = json.load(f)

    print("[DEBUG] Loaded TNR data successfully.")
    return tnr_table, tnr_paths


def tnr_query(source, target, transit_nodes, tnr_table, tnr_paths, G):
    print(f"[DEBUG] Running TNR query from {source} to {target}...")
    if source in transit_nodes and target in transit_nodes:
        return tnr_table.get(source, {}).get(target, float("inf")), tnr_paths.get(
            source, {}
        ).get(target, [])

    nearest_source = min(
        transit_nodes,
        key=lambda t: nx.shortest_path_length(G, source, t, weight="length"),
    )
    nearest_target = min(
        transit_nodes,
        key=lambda t: nx.shortest_path_length(G, target, t, weight="length"),
    )

    first_leg = nx.shortest_path(G, source, nearest_source, weight="length")
    middle_leg = tnr_paths.get(nearest_source, {}).get(nearest_target, [])
    last_leg = nx.shortest_path(G, nearest_target, target, weight="length")

    full_path = first_leg[:-1] + middle_leg + last_leg

    return (
        nx.shortest_path_length(G, source, nearest_source, weight="length")
        + tnr_table.get(nearest_source, {}).get(nearest_target, float("inf"))
        + nx.shortest_path_length(G, nearest_target, target, weight="length"),
        full_path,
    )


def dijkstra_query(G, source, target):
    print(f"[DEBUG] Running Dijkstra query from {source} to {target}...")
    path = nx.shortest_path(G, source, target, weight="length")
    distance = nx.shortest_path_length(G, source, target, weight="length")
    return distance, path


def compare_paths(path1, path2):
    set1, set2 = set(path1), set(path2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0


def compare_algorithms(
    G, transit_nodes, tnr_table, tnr_paths, num_tests=10, error_tolerance=0.05
):
    print("[DEBUG] Starting TNR vs Dijkstra comparison...")
    nodes = list(G.nodes())
    correct_results = 0
    total_jaccard_similarity = 0
    total_tests = 0

    for test_num in range(1, num_tests + 1):
        source, target = random.sample(nodes, 2)

        print(
            f"\n[DEBUG] Test {test_num}/{num_tests} - Source: {source}, Target: {target}"
        )

        dijkstra_dist, dijkstra_path = dijkstra_query(G, source, target)
        tnr_dist, tnr_path = tnr_query(
            source, target, transit_nodes, tnr_table, tnr_paths, G
        )

        if dijkstra_dist == float("inf") or tnr_dist == float("inf"):
            print(
                f"[DEBUG] Skipping test {test_num} - One of the paths is unreachable."
            )
            continue  # Skip disconnected nodes

        error = abs(dijkstra_dist - tnr_dist) / max(
            dijkstra_dist, 1
        )  # Avoid division by zero
        path_similarity = compare_paths(dijkstra_path, tnr_path)

        if error <= error_tolerance:
            correct_results += 1

        total_jaccard_similarity += path_similarity
        total_tests += 1

        print(f"[DEBUG] Test {test_num} Results:")
        print(
            f"  Dijkstra Distance = {dijkstra_dist}, TNR Distance = {tnr_dist}, Error = {error:.2%}"
        )
        print(f"  Path Similarity = {path_similarity:.2%}\n")

    accuracy = (correct_results / total_tests) * 100 if total_tests > 0 else 0
    avg_jaccard_similarity = (
        (total_jaccard_similarity / total_tests) if total_tests > 0 else 0
    )

    print(
        f"[DEBUG] TNR Accuracy within {error_tolerance * 100}% error tolerance: {accuracy:.2f}%"
    )
    print(
        f"[DEBUG] Average Path Similarity (Jaccard Index): {avg_jaccard_similarity:.2f}"
    )
    return accuracy, avg_jaccard_similarity


if __name__ == "__main__":
    print("[DEBUG] Loading graph...")
    G = nx.read_graphml("resources/colorado_springs.graphml")

    print("[DEBUG] Loading transit nodes and TNR data...")
    transit_nodes = load_final_transit_nodes()
    tnr_table, tnr_paths = load_precomputed_tnr_data()

    print("[DEBUG] Starting comparison tests...")
    compare_algorithms(
        G, transit_nodes, tnr_table, tnr_paths, num_tests=1, error_tolerance=0.05
    )
