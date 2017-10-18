import cvxopt
import networkx as nx
import numpy as np
from cvxopt.glpk import ilp
from numpy.linalg import eigh
from pymetis import part_graph
from sklearn.cluster import KMeans

from time import clock
from Jpps import Jpps


class JppsGlpkOjs(Jpps):
    def __init__(self, verbose=False, cut_function="metis"):
        Jpps.__init__(self)
        if not verbose:
            cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'
        if cut_function == "metis":
            self.cut_function = self._metis_cut
        elif cut_function == "spectral":
            self.cut_function = self._spectral_cut

    def solve(self, k, cpu_time=False):
        net_apx = nx.random_geometric_graph(self.graph.order(),
                                            self.jam_radius,
                                            pos=self.pos)
        part_vert = self.cut_function(k)
        G_cut = self.part2cut(self.graph, part_vert)
        G_ext = net_apx.subgraph(list(set.union(*[set(net_apx.neighbors(node)) for node in G_cut.nodes()])))
        Nei = nx.adjacency_matrix(G_ext, nodelist=G_ext.nodes()).todense() + np.diag(np.ones(G_ext.order()))
        Inc = np.transpose(nx.incidence_matrix(G_ext, edgelist=G_cut.edges()).todense())
        ilp_G = cvxopt.matrix(-np.dot(Inc, Nei))

        ilp_h = cvxopt.matrix(-np.ones(len(G_cut.edges())))
        ilp_c = cvxopt.matrix(np.ones(G_ext.order()))

        binvars = set()
        for var in range(G_ext.order()):
            binvars.add(var)
        t_0 = clock()
        status, isol = cvxopt.glpk.ilp(c=ilp_c,
                                       G=ilp_G,
                                       h=ilp_h,
                                       I=binvars,
                                       B=binvars)
        t_opt = clock() - t_0
        nodeMap = {}
        for m, n in enumerate(G_ext.nodes()):
            nodeMap[m] = n

        self.solns[k] = list(map(lambda x: nodeMap[x], list(np.nonzero(isol)[0])))
        if cpu_time:
            return t_opt
        else:
            return

    def place(self, k):
        net_apx = nx.random_geometric_graph(self.graph.order(),
                                            self.jam_radius,
                                            pos=self.pos)
        if k in self.solns:
            jammer = self.solns[k]
        else:
            jammer = self.solve(k)
        jammed = list(set.union(*[set(net_apx.neighbors(node))
                                  for node in jammer]))
        residue_graph = self.graph.copy()
        residue_graph.remove_nodes_from(jammed)
        connected_subgraphs = sorted(nx.connected_components(residue_graph),
                                     key=len,
                                     reverse=True)
        clusters = []
        for i in range(k - 1):
            clusters.append(list(connected_subgraphs[i]))
        clusters.append([])
        for i in range(k - 1, len(connected_subgraphs)):
            clusters[-1] += list(connected_subgraphs[i])
        clusters[-1].sort()

        return {'jammer': jammer,
                'jammed': jammed,
                'clusters': clusters}

    def _metis_cut(self, k):
        _, part_vert = part_graph(k, self.graph)
        return part_vert

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

        L = nx.laplacian_matrix(self.graph).todense()
        l, U = eigh(L)

        eigs = {i: tuple(U[i, j + 1] for j in range(dimension)) for i in range(self.graph.order())}

        X = np.array(list(eigs.values()))
        est = KMeans(k, init='random')
        est.fit(X)
        labels = est.labels_
        return labels

    def part2cut(self, net, part_vert):
        '''
        Generate subgraph induced by edge cut get from vertex
        assignment list

        input: nx graph, list
        output: nx graph
        '''
        G_cut_nodes = set()
        for edge in net.edges():
            if part_vert[edge[0]] != part_vert[edge[1]]:
                G_cut_nodes.add(edge[0])
                G_cut_nodes.add(edge[1])
        return net.subgraph(G_cut_nodes)
