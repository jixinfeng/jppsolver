import networkx as nx
import numpy as np


class Jpps(object):
    DEFAULT_GRAPH_ORDER = 100
    DEFAULT_COMM_RADIUS = 0.15

    def __init__(self):
        self.graph = nx.Graph()
        self.graph_apx = nx.Graph()
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
        self.pos = nx.get_node_attributes(self.graph, 'pos')
        self.comm_radius = comm_radius
        if jam_radius is None:
            self.jam_radius = comm_radius
        else:
            self.jam_radius = jam_radius

    def reset(self):
        self.graph = nx.Graph()
        self.pos = {}
        self.comm_radius = 0
        self.jam_radius = 0

    def solve(self,
              num_cluster,
              circles=None,
              verbose=False,
              num_threads=None,
              cpu_time=False,
              filename=None,
              name="jpp"):
        self.solns[num_cluster] = [0]
        return 0

    def place(self, num_cluster):
        if num_cluster in self.solns:
            jammer = self.solns[num_cluster]
        else:
            jammer = self.solve(num_cluster)
        jammed = list(set.union(*[set(self.graph_apx.neighbors(node))
                                  for node in jammer]) &
                      set(range(self.graph.order())))
        residue_graph = self.graph.copy()
        residue_graph.remove_nodes_from(jammed)
        connected_subgraphs = sorted(nx.connected_components(residue_graph),
                                     key=len,
                                     reverse=True)
        clusters = []
        for i in range(num_cluster - 1):
            clusters.append(list(connected_subgraphs[i]))
        clusters.append([])
        for i in range(num_cluster - 1, len(connected_subgraphs)):
            clusters[-1] += list(connected_subgraphs[i])
        clusters[-1].sort()

        return {'jammer': jammer,
                'jammed': jammed,
                'clusters': clusters}

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
