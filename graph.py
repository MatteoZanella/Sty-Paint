from collections import defaultdict
from copy import copy
import numpy as np
import random

class Node:

    def __init__(self, features):
        self.position = features[:2]
        self.w = features[3]
        self.h = features[4]
        self.color = features[5:11]
        self.alpha = features[-1]

        self.incoming_edges = 0
        self.lam = None

    def area(self):
        return self.h * self.w

class Graph:

    def __init__(self, n_nodes, adjlist, features):
        self.n_nodes = n_nodes
        self.adjlist = adjlist
        self.criterion = 'color'

        # Create  the nodes and assign the features
        self.nodes = defaultdict(list)  # same type as adjlist
        for i in range(self.n_nodes):
            self.nodes[i] = Node(features[0, i, :])

    def counter(self, adjlist):

        output = defaultdict(list)
        # Initialize all to zero
        for i in adjlist.keys():
            output[i] = 0
        for _, adjnodes in adjlist.items():
            for node in adjnodes:
                output[node] += 1
        return output

    def find_zero(self, inc_count):
        output = []
        for node, count in inc_count.items():
            if count == 0:
                # Add node to the list if it doesn't contain incoming edges
                output.append(node)
        return output

    def compute_metric(self, ref, c):
        if ref is None:
            return 0

        flag = False
        pos = np.sum((self.nodes[ref].position - self.nodes[c].position) ** 2)
        col = np.sum((self.nodes[ref].color - self.nodes[c].color) ** 2)
        res = (1 - self.lam) * pos + self.lam * col

        if flag:
            return res
        else:
            return res / (0.5 * self.nodes[c].area())



    def select_next(self, reference, candidates):
        if reference is None:
            random.shuffle(candidates)
            return candidates[0], 0

        scores = []
        for c in candidates:
            score = self.compute_metric(reference, c)
            scores.append(score)

        idx = np.argmin(scores)
        min_score = np.min(scores)
        return candidates[idx], min_score

    def sort(self, lam):
        self.lam = lam
        unvisited_nodes = copy(self.adjlist)
        topo_order = []
        ref = None
        tot_score = 0
        while unvisited_nodes:
            incoming_edges_count = self.counter(unvisited_nodes)
            zero_incoming_edges = self.find_zero(incoming_edges_count)

            # Select next node from zero_incoming_edges based on criterion
            src, ms = self.select_next(reference=ref, candidates=zero_incoming_edges)
            tot_score += ms
            topo_order.append(src)
            unvisited_nodes.pop(src)
            ref = src
        self.lam = None
        return topo_order, tot_score
