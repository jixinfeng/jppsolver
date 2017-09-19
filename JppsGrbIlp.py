import networkx as nx
import numpy as np
import gurobipy as gurobi

from Jpps import Jpps


class JppsGrbIlp(Jpps):
    def __init__(self):
        Jpps.__init__(self)

    def solve(self, ncluster, name="", verbose=False):
        model = gurobi.Model(name)
        model.setParam('OutputFlag', verbose)

        if ncluster in self.solns:
            return

        bk = self.order / ncluster

        net_apx = nx.random_geometric_graph(self.order,
                                            self.jam_radius,
                                            pos=self.pos)

        apx = nx.adj_matrix(G=net_apx,
                            nodelist=range(self.graph.order())) + np.eye(self.graph.order())

        jammer_vars = model.addVars(self.order,
                               lb=0,
                               up=1,
                               vtype=gurobi.GRB.BINARY,
                               name="jammer")

        jammed_vars = model.addVars(self.order,
                               lb=0,
                               up=1,
                               vtype=gurobi.GRB.BINARY,
                               name="jammed")

        cluster_vars = model.addVars(ncluster, self.order,
                                lb=0,
                                up=1,
                                vtype=gurobi.GRB.BINARY,
                                name="x")

        # set objective function
        # minimize the number of jammer needed
        model.setObjective(gurobi.quicksum(jammer_vars), gurobi.GRB.MINIMIZE)

        # add eq (1)
        # each vertex assigned to exactly 1 cluster
        model.addConstrs((cluster_vars.sum("*", i) + jammed_vars[i] == 1 for i in range(self.graph.order())), name="eq1")

        # add ieq (2)
        # at least 1 jammer
        model.addConstr(jammer_vars.sum() >= 1, name="ieq2")

        # add ieq (3)
        # no empty cluster
        model.addConstrs((cluster_vars.sum(k, "*") >= 1 for k in range(ncluster)), name="ieq3")

        # add ieq (4)
        # first cluster should be smaller than ceil(network_order/ncluster)
        model.addConstr(cluster_vars.sum(0, "*") <= bk, name="ieq4")

        # add ieq (5)
        # cluter size should be in non-increasing order
        for k in range(ncluster - 1):
            model.addConstr(lhs=cluster_vars.sum(k, "*"),
                            sense=gurobi.GRB.GREATER_EQUAL,
                            rhs=cluster_vars.sum(k + 1, "*"),
                            name="ieq5[" + str(k) + "]")

        # add ieq (6)
        # nodes close to jammers are jammed
        for i in range(self.order):
            model.addConstr(lhs=jammed_vars[i],
                            sense=gurobi.GRB.LESS_EQUAL,
                            rhs=gurobi.LinExpr(apx[i].tolist()[0], [jammer_vars[j] for j in range(self.order)]),
                            name="ieq6[" + str(i) + "]")

        # add ieq (7)
        # both ends of an edge should stay in same cluster unless one of them is jammed
        for k in range(ncluster):
            for i, j in self.graph.edges():
                model.addConstr(gurobi.LinExpr(cluster_vars[(k, i)] - cluster_vars[(k, j)] - jammed_vars[i] - jammed_vars[j]) <= 0,
                                name="ieq7[" + ",".join(map(str, [i, j, k])) + "]")

        # add ieq (8)
        # both nodes at jammers' locations should be jammed
        model.addConstrs((jammer_vars[i] - jammed_vars[i] <= 0 for i in range(self.order)),
                         name="ieq8")

        model.update()
        model.optimize()
        soln = [i for i, var in enumerate(model.getVars()[:self.graph.order()]) if int(var.x) == 1]
        self.solns[ncluster] = soln

    def place(self, ncluster):
        net_apx = nx.random_geometric_graph(self.order,
                                            self.jam_radius,
                                            pos=self.pos)
        if ncluster in self.solns:
            jammer = self.solns[ncluster]
        else:
            jammer = self.solve(ncluster)
        jammed = list(set.union(*[set(net_apx.neighbors(node))
                                  for node in jammer]))
        residue_graph = self.graph.copy()
        residue_graph.remove_nodes_from(jammed)
        connected_subgraphs = sorted(nx.connected_components(residue_graph),
                                     key=len,
                                     reverse=True)
        clusters = []
        for i in range(ncluster - 1):
            clusters.append(list(connected_subgraphs[i]))
        clusters.append([])
        for i in range(ncluster - 1, len(connected_subgraphs)):
            clusters[-1] += list(connected_subgraphs[i])
        clusters[-1].sort()

        return {'jammer': jammer,
                'jammed': jammed,
                'clusters': clusters}