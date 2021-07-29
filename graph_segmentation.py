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

        self.cl = None

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
            return 0, None

        col = np.sum((self.nodes[ref].color - self.nodes[c].color) ** 2) * 2
        area = (self.nodes[ref].area() - self.nodes[c].area()) ** 2
        cl = (self.nodes[ref].cl != self.nodes[c].cl) * 50
        if self.previous_class is not None:
            cl += (self.previous_class != self.nodes[c].cl) * 0

        res = col + area + cl

        return res, (col, area, cl)

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
        if min_score > 50:
            print(scores)

        return candidates[idx], min_score

    def sort(self):
        self.previous_class = None
        unvisited_nodes = copy(self.adjlist)
        topo_order = []
        ref = None
        tot_score = 0
        while unvisited_nodes:
            incoming_edges_count = self.counter(unvisited_nodes)
            zero_incoming_edges = self.find_zero(incoming_edges_count)

            # Select next node from zero_incoming_edges based on criterion
            src, ms = self.select_next(reference=ref, candidates=zero_incoming_edges)
            _, scores = self.compute_metric(ref, src)
            print(scores)
            print('N {}'.format(len(zero_incoming_edges)))

            topo_order.append(src)
            unvisited_nodes.pop(src)
            if ref is None:
                self.previous_class = None
            else:
                self.previous_class = self.nodes[ref].cl
            ref = src
        self.lam = None
        return topo_order, tot_score
