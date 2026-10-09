"""Linear reachability and one BFS representative path per reachable output."""
from collections import deque
import networkx as nx

def trace_node(ir, node_id, layers=2):
    graph = ir.topology()
    if node_id not in graph:
        raise ValueError(f'未知节点 ID：{node_id}')
    parents = {node_id: None}
    queue = deque([node_id])
    while queue:
        current = queue.popleft()
        for nxt in graph.successors(current):
            if nxt not in parents:
                parents[nxt] = current
                queue.append(nxt)
    nearby = set(nx.single_source_shortest_path_length(graph, node_id, cutoff=layers))
    nearby.update(nx.single_source_shortest_path_length(graph.reverse(copy=False), node_id, cutoff=layers))
    paths, reachable, omitted = [], [], False
    for output in ir.model['outputs']:
        target = output['producer']
        if target not in parents:
            continue
        reachable.append(output['name'])
        path, current = [], target
        while current is not None:
            path.append(current)
            current = parents[current]
        path.reverse()
        edges = []
        for source, target in zip(path, path[1:]):
            edges.append({'source': source, 'target': target, 'tensors': graph[source][target]['tensors']})
        paths.append({'output_tensor': output['name'], 'nodes': path, 'edges': edges})
    # Compute path multiplicity saturated at 2, never enumerate alternative paths.
    counts = {node_id: 1}
    if nx.is_directed_acyclic_graph(graph):
        for current in nx.topological_sort(graph):
            if current in counts:
                for nxt in graph.successors(current):
                    counts[nxt] = min(2, counts.get(nxt, 0) + counts[current])
        omitted = any(counts.get(o['producer'], 0) > 1 for o in ir.model['outputs'] if o['name'] in reachable)
    else:
        omitted = True
    return {'node_id': node_id, 'predecessors': sorted(graph.predecessors(node_id)),
        'successors': sorted(graph.successors(node_id)), 'upstream_neighbors': sorted(set(nx.single_source_shortest_path_length(graph.reverse(copy=False), node_id, cutoff=layers)) - {node_id}),
        'neighborhood': sorted(nearby), 'downstream_nodes': sorted(set(parents) - {node_id}),
        'reachable_outputs': reachable, 'representative_paths': paths, 'paths_truncated': omitted,
        'path_policy': '每个可达输出仅展示一条 BFS 最短代表路径；其他路径不枚举',
        'reason': '依赖关系不等于下游违规、CPU 回退或精度下降' if reachable else '未发现到模型输出的依赖路径，可能为死分支'}
