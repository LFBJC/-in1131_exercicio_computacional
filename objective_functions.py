import numpy as np
from pymoo.core.problem import Problem

class Ackley(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=-32.768,
                         xu=32.768)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = -20*np.exp(-0.2*np.sqrt(np.mean(x**2)))-np.exp(np.mean(np.cos(2*np.pi*x)))+20+np.e
        out["F"] = f1

