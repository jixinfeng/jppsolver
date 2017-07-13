import cvxopt
from cvxopt.glpk import ilp

from JppSolver import JppSolver


class JppSolverOjs(JppSolver):
    def __init__(self, verbose=False):
        JppSolver.__init__()
        if not verbose:
            cvxopt.glpk.options['msg_lev'] = 'GLP_MSG_OFF'

    def solve(self, k):
        pass
