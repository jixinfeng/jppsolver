from jppsolver.JppsGrbIlp import *
from network100_1 import sample_pos_100
from network100_1_circles import sample_pos_100_circle

jpp = JppsGrbIlp()
jpp.set_graph_via_pos(sample_pos_100, 0.15)
print(jpp.solve(4, circles=sample_pos_100_circle, verbose=True, cpu_time=True))
print(jpp.place(4))
