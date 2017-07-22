import networkx as nx
import numpy as np


class JppSolver(object):
    DEFAULT_GRAPH_ORDER = 100
    DEFAULT_COMM_RADIUS = 0.15

    def __init__(self):
        self.graph = nx.Graph()
        self.order = 0
        self.pos = {}
        self.comm_radius = 0
        self.jam_radius = 0
        self.solns = {}

    def set_graph(self,
                  graph,
                  comm_radius,
                  jam_radius=None):
        if not nx.is_connected(graph):
            raise AttributeError("Graph not connected!")
        self.reset()
        self.graph = graph
        self.order = graph.order()
        self.pos = nx.get_node_attributes(graph, 'pos')
        if graph.order() != len(self.pos):
            raise AttributeError("Node position does not match!")
        self.comm_radius = comm_radius
        if jam_radius is None:
            self.jam_radius = comm_radius
        else:
            self.jam_radius = jam_radius

    def set_graph_via_pos(self,
                          pos,
                          comm_radius,
                          jam_radius=None):
        if jam_radius is None:
            jam_radius = comm_radius
        graph = nx.random_geometric_graph(n=len(pos),
                                          radius=comm_radius,
                                          pos=pos)
        self.set_graph(graph, comm_radius, jam_radius)

    def set_random_graph(self,
                         order=100,
                         comm_radius=0.15,
                         jam_radius=None):
        self.reset()
        self.graph = self._generate_connected_graph(order, comm_radius)
        self.order = order
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        self.comm_radius = comm_radius
        if jam_radius is None:
            self.jam_radius = comm_radius
        else:
            self.jam_radius = jam_radius

    def solve(self, k):
        return []

    def place(self):
        return {'jammer': [],
                'jammed': [],
                'clusters': []}

    def reset(self):
        self.graph = nx.Graph()
        self.order = 0
        self.pos = {}
        self.comm_radius = 0
        self.jam_radius = 0
        self.solns = {}

    def _generate_connected_graph(self, order, comm_radius):
        mag_ratio = np.sqrt(order / self.DEFAULT_GRAPH_ORDER)
        if comm_radius < self.DEFAULT_COMM_RADIUS:
            print("WARNING: this order, comm_radius combination may cause unpredictable long running time")

        graph = nx.random_geometric_graph(order, comm_radius / mag_ratio)
        while not nx.is_connected(graph):
            graph = nx.random_geometric_graph(order, comm_radius / mag_ratio)

        pos = nx.get_node_attributes(graph, 'pos')
        for i in range(order):
            pos[i] = tuple(j * mag_ratio for j in pos[i])

        return nx.random_geometric_graph(n=order,
                                         radius=comm_radius,
                                         pos=pos)
