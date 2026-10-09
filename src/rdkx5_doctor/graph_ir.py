"""Tensor-labelled graph; node order is identity, never connectivity."""
from dataclasses import dataclass, field
import networkx as nx

@dataclass
class GraphIR:
    model: dict
    nodes: list[dict]
    tensors: dict[str, dict]
    edges: list[dict]
    warnings: list[str]
    small_constants: dict = field(default_factory=dict)

    numeric_constants: dict = field(default_factory=dict)

    def topology(self):
        graph = nx.DiGraph()
        graph.add_nodes_from(n['id'] for n in self.nodes)
        for edge in self.edges:
            source, target = edge['source'], edge['target']
            if graph.has_edge(source, target):
                graph[source][target]['tensors'].append(edge['tensor'])
            else:
                graph.add_edge(source, target, tensors=[edge['tensor']])
        return graph
