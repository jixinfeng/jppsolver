from JppSolverIlp import *
from JppSolverOjs import *
from network100_1 import sample_pos_100


jpp = JppSolverIlp(verbose=True)
#jpp = JppSolverOjs(verbose=True)
jpp.set_graph_via_pos(sample_pos_100, 0.15)
print(jpp.solve(2))
print(jpp.place(2))
