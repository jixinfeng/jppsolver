import cvxopt
import networkx as nx
import numpy as np
from cvxopt.glpk import ilp
from scipy.sparse import coo_matrix

from Jpps import Jpps


class JppsGlpkIlp(Jpps):
    def __init__(self, verbose=False):
        Jpps.__init__(self)
        if not verbose:
            cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

    def solve(self, k):
        if k in self.solns:
            return
        bk = self.order / k

        N = self.graph.number_of_nodes()
        E = self.graph.number_of_edges()

        A = coo_matrix(self._makea(N, k))
        b = self._makeb(N)
        G = coo_matrix(self._makeg(N, k, E))
        h = self._makeh(N, k, E, bk)
        c = self._makec(N, k)

        binvars = set()
        for var in range((k + 2) * N):
            binvars.add(var)

        status, isol = ilp(c=cvxopt.matrix(c),
                           G=cvxopt.spmatrix(G.data,
                                             G.row,
                                             G.col,
                                             G.shape),
                           h=cvxopt.matrix(h),
                           A=cvxopt.spmatrix(A.data,
                                             A.row,
                                             A.col,
                                             A.shape),
                           b=cvxopt.matrix(b),
                           I=binvars,
                           B=binvars)
        self.solns[k] = self._parse(isol)
        return self.solns[k]

    def place(self, k):
        net_apx = nx.random_geometric_graph(self.order,
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

    def _parse(self, isol):
        soln = []
        for i, j in enumerate(isol[:self.order]):
            if int(j) == 1:
                soln.append(i)
        return soln

    def _singlerow(self, N, K, i):
        """
        creat a matrix with exactly one row of 1s and others are 0
        """
        mat = np.zeros([K, N])
        mat[i] = np.ones([N])
        return mat

    def _singlerow1(self, N, K, i):
        """
        creat one row matrix to sum indicators of one cluster
        """
        mat = np.zeros([1, (K + 2) * N])
        mat[0, (i + 1) * N: (i + 2) * N] = np.ones([N])
        return mat

    def _singlerow2(self, N, K, i, j):
        """
        i >= 1
        j > i >= 2
        creat one row matrix to compare size of two clusters
        """
        mat = np.zeros([1, (K + 2) * N])
        mat[0, (i + 1) * N: (i + 2) * N] = -np.ones([N])
        mat[0, (j + 1) * N: (j + 2) * N] = np.ones([N])
        return mat

    def _makeb1(self, inc):
        """
        building block 1
        """
        return np.transpose(-inc)

    def _makeb2(self, inc, N, E):
        """
        building block 2
        """
        B2 = np.transpose(-inc)
        for i in range(E):
            for j in range(N):
                if B2[i][j] != 0:
                    B2[i][j] = -B2[i][j]
                    break
        return B2

    def _g4row(self, B1, B2, N, E, K, j):
        """
        one row of contrain 8
        """
        mat = np.concatenate((np.zeros([E, N]), B1), axis=1)
        for i in range(K):
            if i != j:
                mat = np.concatenate((mat, np.zeros([E, N])), axis=1)
            else:
                mat = np.concatenate((mat, B2), axis=1)
        return mat

    def _makea(self, N, K):
        """
        create equality constrains Ax = b (1)
        every node should be 1 and only 1 cluster
        N rows
        """
        A = np.zeros([N, N])
        for i in range(K + 1):
            A = np.concatenate((A, np.eye(N)), axis=1)
        return A

    def _makeb(self, N):
        """
        N rows
        """
        b = np.ones(N)
        return b

    def _makeg1(self, N, K):
        """
        create constrain (2)
        there should be at least one jammer
        1 row
        """
        mat = np.zeros([1, (K + 2) * N])
        for i in range(N):
            mat[0][i] = -1
        return mat

    def _makeg2(self, N, K):
        """
        create inequality constrain (3)
        size of each cluster should be at least 1
        K rows
        """
        mat = np.zeros([K, 2 * N])
        for i in range(K):
            mat = np.concatenate((mat, -self._singlerow(N, K, i)), axis=1)
        return mat

    def _makeg3(self, N, K):
        """
        create inequality constrain (4, 5)
        size of biggest cluster is smaller than bk and size of clusters are sorted
        K rows
        """
        mat = self._singlerow1(N, K, 1)
        for i in range(1, K):
            mat = np.concatenate((mat, self._singlerow2(N, K, i, i + 1)), axis=0)
        return mat

    def _makeg4(self, N, K, apx):
        """
        create inquality constrain (6)
        if a node is jammed, it should be within the range of a jammer
        N rows
        """
        mat = np.concatenate((-apx, np.eye(N)), axis=1)
        for i in range(K):
            mat = np.concatenate((mat, np.zeros([N, N])), axis=1)
        return mat

    def _makeg5(self, N, K, E, inc):
        """
        create constrain (8)
        nodes in different clusters shouldn't be connected unless one is in cluster #0
        E * K rows
        """
        B1 = self._makeb1(inc)
        B2 = self._makeb2(inc, N, E)

        mat = self._g4row(B1, B2, N, E, K, 0)
        for i in range(K - 1):
            mat = np.concatenate((mat, self._g4row(B1, B2, N, E, K, i + 1)), axis=0)
        return mat

    def _makeg6(self, N, K):
        """
        create constrain (9, 10)
        all variables must be positive
        (K + 2) * N rows
        """
        mat = -np.eye((K + 2) * N)
        return mat

    def _makeg7(self, N, K):
        """
        create constrain (11)
        jammers themselves must be jammed => in cluster #0
        N rows
        """
        mat = np.concatenate((np.eye(N), -np.eye(N)), axis=1)
        for i in range(K):
            mat = np.concatenate((mat, np.zeros([N, N])), axis=1)
        return mat

    def _makeg(self, N, K, E):
        """
        creat G matrix for inequality constrains

        (K + 4) * N + 2 * K + 2 * E + 1 rows
        """
        adj = np.array(nx.adjacency_matrix(G=self.graph,
                                           nodelist=list(range(N))).todense()) + np.eye(N)
        inc = np.array(nx.incidence_matrix(G=self.graph,
                                           nodelist=list(range(N))).todense())

        net_apx = nx.random_geometric_graph(self.order,
                                            self.jam_radius,
                                            pos=self.pos)
        apx = np.array(nx.adjacency_matrix(G=net_apx,
                                           nodelist=list(range(N))).todense()) + np.eye(N)
        G = np.concatenate((self._makeg1(N, K),
                            self._makeg2(N, K),
                            self._makeg3(N, K),
                            self._makeg4(N, K, apx),
                            self._makeg5(N, K, E, inc),
                            self._makeg6(N, K),
                            self._makeg7(N, K)),
                           axis=0)
        return G

    def _makeh(self, N, K, E, bk):
        h = np.zeros((E + 2) * K + (K + 4) * N + 1)
        for i in range(K + 1):
            h[i] = -1
        h[K + 1] = bk
        return h

    def _makec(self, N, K):
        c = np.zeros((K + 2) * N)
        for i in range(N):
            c[i] = 1
        return c
