import networkx as nx
import numpy as np
import gurobipy as gurobi

from sklearn.cluster import KMeans
from numpy.linalg import eigh

from psutil import cpu_count
from time import clock
from Jpps import Jpps


class JppsGrbIlp(Jpps):
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
        apx = nx.adj_matrix(G=self.graph_apx,
                            nodelist=range(self.graph_apx.order())) + np.eye(self.graph_apx.order())

        part_vert = self._spectral_cut(self.graph, num_cluster)
        G_cut = self._part_to_cut(self.graph, part_vert)
        G_ext = self.graph_apx.subgraph(list(set.union(*[set(self.graph_apx.neighbors(node))
                                                         for node in G_cut.nodes()])))
        Nei = nx.adjacency_matrix(G_ext, nodelist=G_ext.nodes()).todense() + np.diag(np.ones(G_ext.order()))
        Inc = np.transpose(nx.incidence_matrix(G_ext, edgelist=G_cut.edges()).todense())

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
        t_0 = clock()

        if filename:
            model.write(filename)

        model.optimize()
        t_opt = clock() - t_0
        soln = [node_map[i] for i, var in enumerate(model.getVars()) if int(var.x) == 1]
        self.solns[num_cluster] = soln

        if cpu_time:
            return t_opt
        else:
            return

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

        L = nx.normalized_laplacian_matrix(self.graph)
        l, U = eigh(L)

        eigs = {i: tuple(U[i, j + 1] for j in range(dimension)) for i in range(self.graph.order())}

        X = np.array(list(eigs.values()))
        est = KMeans(k, init='random')
        est.fit(X)
        labels = est.labels_
        return labels

    @staticmethod
    def _part_to_cut(net, part_vert):
        """
        Generate subgraph induced by edge cut get from vertex
        assignment list

        input: nx graph, list
        output: nx graph
        """
        G_cut_nodes = set()
        for edge in net.edges():
            if part_vert[edge[0]] != part_vert[edge[1]]:
                G_cut_nodes.add(edge[0])
                G_cut_nodes.add(edge[1])
        return net.subgraph(G_cut_nodes)
