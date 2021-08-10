from collections import defaultdict
from copy import copy
import numpy as np


def counter(adjlist):
    output = defaultdict(list)
    # Initialize all to zero
    for i in adjlist.keys():
        output[i] = 0
    for _, adjnodes in adjlist.items():
        for node in adjnodes:
            output[node] += 1
    return output

def find_zero(inc_count):
    output = []
    for node, count in inc_count.items():
        if count == 0:
            # Add node to the list if it doesn't contain incoming edges
            output.append(node)
    return output


class Node:

    def __init__(self, features):
        self.position = features[:2]
        self.w = features[3]
        self.h = features[4]
        self.color = features[5:11]
        self.alpha = features[11]

        self.cl = features[12]
        self.is_salient = features[13]

    def area(self):
        return self.h * self.w


class Graph:

    def __init__(self, adjlist, features):
        self.n_nodes = len(adjlist)
        self.adjlist = adjlist

        # Create  the nodes and assign the features
        self.nodes = defaultdict(list)  # same type as adjlist
        for i in range(self.n_nodes):
            self.nodes[i] = Node(features[0, i, :])

    def set_weights(self, w_dict):
        self.w_color = w_dict['color']
        self.w_area =  w_dict['area']
        self.w_pos = w_dict['pos']
        self.w_class =  w_dict['class']
        self.w_saliency = w_dict['sal']


    def reset_weights(self):
        self.w_color = None
        self.w_area =  None
        self.w_pos = None
        self.w_class =  None
        self.w_saliency = None


    def compute_metric(self, ref, c):
        """
        LKH
        """

        color = np.sum((self.nodes[ref].color - self.nodes[c].color) ** 2)
        area = (self.nodes[ref].area() - self.nodes[c].area()) ** 2
        pos = np.sum((self.nodes[ref].position - self.nodes[c].position) ** 2)
        cl = (self.nodes[ref].cl != self.nodes[c].cl)
        sal = (1 - self.nodes[c].is_salient)

        cost = self.w_color * color + \
               self.w_area * area + \
               self.w_pos * pos + \
               self.w_class * cl + \
               self.w_saliency * sal

        return cost

    def starting_node(self):
        """
        This can be used also by LKH
        """
        incoming_edges_count = counter(self.adjlist)
        zero_incoming_edges = find_zero(incoming_edges_count)

        scores = []
        for c in zero_incoming_edges:
            scores.append(-len(self.adjlist[c]))

        idx = np.argmin(scores)
        return zero_incoming_edges[idx]

    def select_next(self, reference, candidates):
        scores = []
        for c in candidates:
            score = self.compute_metric(reference, c)
            scores.append(score)

        idx = np.argmin(scores)
        min_score = np.min(scores)

        return candidates[idx], min_score

    def sort(self):
        topo_order = []
        unvisited_nodes = copy(self.adjlist)


        ref = self.starting_node()
        topo_order.append(ref)
        unvisited_nodes.pop(ref)

        while unvisited_nodes:
            incoming_edges_count = counter(unvisited_nodes)
            zero_incoming_edges = find_zero(incoming_edges_count)

            # Select next node from zero_incoming_edges based on criterion
            src, _ = self.select_next(reference=ref, candidates=zero_incoming_edges)

            topo_order.append(src)
            unvisited_nodes.pop(src)
            ref = src

        return topo_order

class GraphBuilder:

    def __init__(self, transparency, hidden=True):
        self.transparency = transparency
        self.n = transparency.shape[0]
        self.hidden = hidden

    def build_graph(self):

        adj_list = defaultdict(list)

        for adj_id in range(self.n):
            #print('{} / {}'.format(adj_id, self.n))
            curr = self.transparency[adj_id]
            next_strokes = self.transparency[adj_id + 1:]
            overlap_area = np.logical_and(curr, next_strokes)
            overlap_id = np.nonzero(overlap_area.sum(axis=(1, 2)))[0]

            if self.hidden:
                to_remove = self.unimportant_overlaps(overlap_area, overlap_id, adj_id+1)
            else:
                to_remove = []

            adj_list[adj_id] = {}
            adj_list[adj_id]['to_remove'] = to_remove
            adj_list[adj_id]['all_edges'] = overlap_id + (adj_id+1)

        return adj_list

    def unimportant_overlaps(self, overlap_area, overlap_id, base_id):
        """
        If an overlap is later covered by another storke, than it can be ignored.
        """
        to_remove = []
        for j in range(len(overlap_id)):
            ref_id = overlap_id[j]
            for k in range(j + 1, len(overlap_id)):  # check only next strokes
                curr_id = overlap_id[k] + base_id  # original index
                if np.logical_and(overlap_area[ref_id], self.transparency[curr_id]).sum() / overlap_area[
                    ref_id].sum() > 0.99:
                    to_remove.append(ref_id + base_id)
                    break
        return to_remove

