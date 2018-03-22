import networkx as nx
import numpy as np
import gurobipy as gurobi

from sklearn.cluster import KMeans
from numpy.linalg import eigh

from os import cpu_count
from time import clock
from .Jpps import Jpps


class JppsGrbOjs(Jpps):
    def __init__(self):
        Jpps.__init__(self)
        self.graph_apx = None

    def solve(self,
              num_cluster,
              circles=None,
              verbose=False,
              num_threads=None,
              cpu_time=False,
              filename=None,
              name="jpp"):

        model = gurobi.Model(name)
        model.setParam('OutputFlag', verbose)
        if num_threads:
            model.setParam("Threads", num_threads)
        else:
            model.setParam("Threads", cpu_count())

        if num_cluster in self.solns:
            return

        pos_apx = self.pos.copy()
        if circles:
            for circle in sorted(circles.values()):
                pos_apx[len(pos_apx)] = [circle[0], circle[1]]

        self.graph_apx = nx.random_geometric_graph(len(pos_apx),
                                                   self.jam_radius,
                                                   pos=pos_apx)

        part_vert = self._spectral_cut(num_cluster)
        G_cut = self._part_to_cut(part_vert)
        G_ext = self.graph_apx.subgraph(list(set.union(*[set(self.graph_apx.neighbors(node))
                                                         for node in G_cut.nodes()])))

        Nei = nx.adjacency_matrix(G_ext, nodelist=G_ext.nodes()).todense() + np.diag(np.ones(G_ext.order()))

        node_map = {}
        node_inverse_map = {}
        for m, n in enumerate(G_ext.nodes()):
            node_map[m] = n
            node_inverse_map[n] = m

        jammer_vars = model.addVars(self.graph_apx.order(),
                                    lb=0,
                                    up=1,
                                    vtype=gurobi.GRB.BINARY,
                                    name="jammer")

        # set objective function
        # minimize the number of jammer needed
        model.setObjective(gurobi.quicksum(jammer_vars), gurobi.GRB.MINIMIZE)

        # add ieq (1)
        # at least 1 jammer
        model.addConstr(jammer_vars.sum() >= 1, name="ieq1")

        # add ieq (2)
        # each edge in E_s has at least one endpoint covered by a jammer
        model.addConstrs((gurobi.LinExpr([(Nei[node_inverse_map[p], i] + Nei[node_inverse_map[q], i], jammer_vars[i])
                                          for i in range(G_ext.order())]) >= 1
                          for p, q in G_cut.edges()), name="ieq2")

        model.update()

        if filename:
            model.write(filename)

        t_0 = clock()
        model.optimize()
        t_opt = clock() - t_0

        soln = [node_map[i] for i, var in enumerate(model.getVars()) if int(var.x) == 1]

        assert model.status == 2

        self.solns[num_cluster] = soln

        if cpu_time:
            return t_opt
        else:
            return model.Runtime

    def _spectral_cut(self, k, dimension=None):
        """
        Cluster the given graph into k clusters using k-means
        via Lloyd's algorithm

        input: nx graph, int
        """
        if dimension is None:
            dimension = k - 1
        if dimension >= self.graph.order() - 1:
            dimension = self.graph.order()

        lap = nx.normalized_laplacian_matrix(self.graph).todense()
        e_values, e_vectors = eigh(lap)

        # eigs = {i: tuple(e_vectors[i, j + 1]
        #                  for j in range(dimension))
        #         for i in range(self.graph.order())}
        # X = np.array([e_coords[k] for k in range(len(e_coords))])

        e_coordinates = np.array([[e_vectors[i, j + 1]
                                   for j in range(dimension)]
                                  for i in range(self.graph.order())])

        est = KMeans(k, init='random')
        est.fit(e_coordinates)
        labels = est.labels_
        return labels

    def _part_to_cut(self, part_vert):
        """
        Generate subgraph induced by edge cut get from vertex
        assignment list

        input: nx graph, list
        output: nx graph
        """
        G_cut_nodes = set()
        for edge in self.graph.edges():
            if part_vert[edge[0]] != part_vert[edge[1]]:
                G_cut_nodes.add(edge[0])
                G_cut_nodes.add(edge[1])
        return self.graph.subgraph(G_cut_nodes)
