from collections import defaultdict
from copy import copy
import numpy as np
import random


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
        self.alpha = features[-1]

        self.cl = None

    def area(self):
        return self.h * self.w


class Graph:

    def __init__(self, adjlist, features):
        self.n_nodes = len(adjlist)
        self.adjlist = adjlist
        self.criterion = 'color'

        # Create  the nodes and assign the features
        self.nodes = defaultdict(list)  # same type as adjlist
        for i in range(self.n_nodes):
            self.nodes[i] = Node(features[0, i, :])

    def compute_metric(self, ref, c):
        if ref is None:
            return 0, None

        col = np.sum((self.nodes[ref].color - self.nodes[c].color) ** 2)
        area = (self.nodes[ref].area() - self.nodes[c].area()) ** 2
        cl = (self.nodes[ref].cl != self.nodes[c].cl) * 10

        res = col + area + cl

        return res, (col, area, cl)

    def lkh(self, i, j):
        col = np.sum((self.nodes[i].color - self.nodes[j].color) ** 2) * 2
        area = (self.nodes[i].area() - self.nodes[j].area()) ** 2
        pos = np.sum((self.nodes[i].position - self.nodes[j].position) ** 2) * 2

        return col+area+pos

    def select_next(self, reference, candidates):
        if reference is None:
            # Choose as starting point a big object and a big stroke
            scores = []
            for c in candidates:
                scores.append(-len(self.adjlist[c]))
        else:
            scores = []
            for c in candidates:
                score, _ = self.compute_metric(reference, c)
                scores.append(score)

        idx = np.argmin(scores)
        min_score = np.min(scores)

        return candidates[idx], min_score

    def sort(self):
        self.previous_class = None
        unvisited_nodes = copy(self.adjlist)
        topo_order = []
        ref = None
        while unvisited_nodes:
            incoming_edges_count = counter(unvisited_nodes)
            zero_incoming_edges = find_zero(incoming_edges_count)

            # Select next node from zero_incoming_edges based on criterion
            src, ms = self.select_next(reference=ref, candidates=zero_incoming_edges)

            topo_order.append(src)
            unvisited_nodes.pop(src)
            ref = src

        return topo_order



# ---- ## --

class GraphBuilder:

    def __init__(self, transparency, treshold=0, hidden=True):
        self.transparency = transparency
        self.treshold = treshold
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
