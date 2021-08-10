from collections import defaultdict
import numpy as np


def dfs(visited, graph, node):
    if node not in visited:
        visited.add(node)
        for neighbour in graph[node]:
            dfs(visited, graph, neighbour)

    return visited


def dfs_paths(graph):
    out = {}

    for node in graph.keys():
        path = set()
        path = dfs(path, graph, node)
        path = list(path)
        path.remove(node)
        path.sort()
        out[node] = path

    return out

def clear_adj(x, hidden=False):
    '''
    hidden = True remove the to_remove nodes from the adj list
    '''
    out = defaultdict(list)

    for k, elem in x.items():
        if hidden:
            out[k] = [e for e in elem['all_edges'] if e not in elem['to_remove']]
        else:
            out[k] = [e for e in elem['all_edges']]

    return out


def lkh_cost_matrix(graph, start):
    # Create the cost matrix of size (n+1) x (n+1)
    C = np.zeros((graph.n_nodes + 1, graph.n_nodes + 1))

    # Populte the matrix
    for i in range(graph.n_nodes):
        for j in range(i + 1, graph.n_nodes):
            # Keep the convention of the documentation
            C[j, i] = graph.compute_metric(i, j)

    # The matrix is symetric
    C = C + np.transpose(C)

    # LKH requires integers, keep 3 decimals (maybe change this)
    C = np.rint(C * 1000)

    # Precedence
    for i, elems in graph.adjlist.items():
        for j in elems:
            C[j, i] = -1  # node i comes before j
            C[i, j] = 0

    # Swap rows and columns with the element in position start
    C[:, [0, start]] = C[:, [start, 0]]
    C[[0, start], :] = C[[start, 0], :]

    # Other requirements by lkh solver
    C[0, :-1] = 0
    C[0, -1] = 1000000  # Huge cost between first and last, from an example
    C[1:, 0] = -1  # First node comes before all the others
    C[-1, :-1] = -1  # Last node comes after all the others

    return C