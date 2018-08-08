import networkx as nx
import numpy as np
import gurobipy as gurobi

from os import cpu_count
from time import process_time
from .Jpps import Jpps


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

        bk = self.graph.order() / num_cluster

        pos_apx = self.pos.copy()
        if circles:
            for circle in sorted(circles.values()):
                pos_apx[len(pos_apx)] = [circle[0], circle[1]]

        self.graph_apx = nx.random_geometric_graph(
            len(pos_apx),
            self.jam_radius,
            pos=pos_apx
        )

        if circles:
            apx = nx.adj_matrix(
                G=self.graph_apx,
                nodelist=range(self.graph_apx.order())
            )
        else:
            apx = nx.adj_matrix(
                G=self.graph_apx,
                nodelist=range(self.graph_apx.order())
            ) + np.eye(self.graph_apx.order())

        # vector x
        if circles:
            jammer_vars = model.addVars(
                len(circles),
                lb=0,
                up=1,
                vtype=gurobi.GRB.BINARY,
                name="jammer"
            )
        else:
            jammer_vars = model.addVars(
                self.graph_apx.order(),
                lb=0,
                up=1,
                vtype=gurobi.GRB.BINARY,
                name="jammer"
            )
        # vector y[0]
        jammed_vars = model.addVars(
            self.graph.order(),
            lb=0,
            up=1,
            vtype=gurobi.GRB.BINARY,
            name="jammed"
        )

        # vector y[1] -- y[K]
        cluster_vars = model.addVars(
            num_cluster, self.graph.order(),
            lb=0,
            up=1,
            vtype=gurobi.GRB.BINARY,
            name="x"
        )

        # set objective function
        # minimize the number of jammer needed
        model.setObjective(gurobi.quicksum(jammer_vars), gurobi.GRB.MINIMIZE)

        # add eq (1)
        # each vertex assigned to exactly 1 cluster
        model.addConstrs((cluster_vars.sum("*", i) + jammed_vars[i] == 1
                          for i in range(self.graph.order())),
                         name="eq1")

        # add ieq (2)
        # at least 1 jammer
        model.addConstr(jammer_vars.sum() >= 1, name="ieq2")

        # add ieq (3)
        # no empty cluster
        model.addConstrs((cluster_vars.sum(k, "*") >= 1 for k in range(num_cluster)), name="ieq3")

        # add ieq (4)
        # first cluster should be smaller than ceil(network_order/num_cluster)
        model.addConstr(cluster_vars.sum(0, "*") <= bk, name="ieq4")

        # add ieq (5)
        # cluter size should be in non-increasing order
        for k in range(num_cluster - 1):
            model.addConstr(lhs=cluster_vars.sum(k, "*"),
                            sense=gurobi.GRB.GREATER_EQUAL,
                            rhs=cluster_vars.sum(k + 1, "*"),
                            name="ieq5[" + str(k) + "]")

        # add ieq (6)
        # nodes close to jammers are jammed
        if circles:
            for i in range(self.graph.order()):
                model.addConstr(lhs=jammed_vars[i],
                                sense=gurobi.GRB.LESS_EQUAL,
                                rhs=gurobi.LinExpr(apx[i].todense().tolist()[0][self.graph.order():],
                                                   [jammer_vars[j] for j in range(len(circles))]),
                                name="ieq6[" + str(i) + "]")
        else:
            for i in range(self.graph.order()):
                model.addConstr(lhs=jammed_vars[i],
                                sense=gurobi.GRB.LESS_EQUAL,
                                rhs=gurobi.LinExpr(apx[i].tolist()[0],
                                                   [jammer_vars[j] for j in range(self.graph_apx.order())]),
                                name="ieq6[" + str(i) + "]")

        # add ieq (7)
        # both ends of an edge should stay in same cluster unless one of them is jammed
        for k in range(num_cluster):
            for i, j in self.graph.edges():
                model.addConstr(gurobi.LinExpr(cluster_vars[(k, i)] - cluster_vars[(k, j)] -
                                               jammed_vars[i] - jammed_vars[j]) <= 0,
                                name="ieq7[" + ",".join(map(str, [i, j, k])) + "]")

        # # add ieq (8)
        # # both nodes at jammers' locations should be jammed
        # # this contraint is actually redundant with the apx matrix defined earlier in this file
        # model.addConstrs((jammer_vars[i] - jammed_vars[i] <= 0 for i in range(self.graph.order())),
        #                  name="ieq8")

        model.update()

        if filename:
            model.write(filename)

        t_0 = process_time()
        model.optimize()
        t_opt = process_time() - t_0

        assert model.status == 2

        soln = [i for i, var in enumerate(model.getVars()[:len(jammer_vars)]) if int(var.x) == 1]
        # WITH SJCs
        # element in solution means the index of disk in SJCs
        # which is the `i` in
        # for i, circle in enumerate(sorted(circles_dict.values())

        # WITHOUT SJCs
        # element in solution means the index of node in G
        # which is the `i` in
        # for i in G.nodes()
        self.solns[num_cluster] = soln

        if cpu_time:
            return t_opt
        else:
            return model.Runtime
