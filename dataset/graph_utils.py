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


def compute_total_cost(graph, tour):
    i = tour[0]
    cost = 0
    for j in tour[1:]:
        cost += graph.compute_metric(i,j)
        i = j

    return cost

def check_correctness(graph, tour):
    # Check
    all_good = True
    for i in range(len(tour)):
        curr = tour[i]
        following = tour[i + 1:]


        for x in following:
            if curr in graph.adjlist[x]:
                all_good = False
                print('Following: {} should be before: {}'.format(x, curr))

    if all_good:
        print('--> Precedence Constraints are satisfied')





def lkh_cost_matrix(graph, start):
    scale = 100
    C = np.zeros((graph.n_nodes + 1, graph.n_nodes + 1))

    idx = np.arange(graph.n_nodes)
    idx[0] = start
    idx[start] = 0

    # Populte the matrix
    for i in range(len(idx)):
        for j in range(len(idx)):
            # Keep the convention of the documentation
            if i == j:
                C[j, i] = 0
            else:
                C[j, i] = graph.compute_metric(idx[i], idx[j])

    # scale
    C = C * scale
    C[0, -1] = 32767
    C = np.rint(C).astype('int16')

    # Other requirements by lkh solver
    C[1:, 0] = -1  # First node comes before all the others
    C[-1, :-1] = -1  # Last node comes after all the others

    for i, adj_i in graph.adjlist.items():
        for j in adj_i:
            C[j, i] = -1

    return C