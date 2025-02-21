import osmnx as ox
import networkx as nx


class Djikstra:
    def __init__(self, graph, source, target):
        self.graph = graph
        self.source = source
        self.target = target
        self.distances = {node: float("inf") for node in self.graph.nodes}
        self.distances[source] = 0
        self.priority_queue = [(0, source)]
        self.predecessors = {}
