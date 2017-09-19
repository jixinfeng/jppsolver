from JppsGlpkIlp import *
from JppsGlpkOjs import *
from network100_1 import sample_pos_100


jpp = JppsGlpkIlp(verbose=True)
jpp.set_graph_via_pos(sample_pos_100, 0.15)
print(jpp.solve(2))
print(jpp.place(2))

jpp = JppsGlpkOjs(verbose=True)
jpp.set_graph_via_pos(sample_pos_100, 0.15)
print(jpp.solve(2))
print(jpp.place(2))
