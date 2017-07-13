from JppSolverIlp import *
from network100_1 import sample_pos_100


jpp = JppSolverIlp(verbose=True)
jpp.set_graph_via_pos(sample_pos_100, 0.15)
print(jpp.solve(2))