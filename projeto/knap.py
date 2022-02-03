import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
import autograd.numpy as anp

class KnapSack(Problem):
    def __init__(self, knapsacks, items):
        self.items = items
        self.knapsacks = knapsacks
        self.name = 'KnapSack'
        super().__init__(n_var=len(knapsacks) * len(items),
                         n_obj=1,
                         n_constr=len(knapsacks),
                         xl=0,
                         xu=1,
                         type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = -1*np.sum([
            item[1]*x[:, j*len(self.items)+i]
            for i, item in enumerate(self.items)
            for j, _ in enumerate(self.knapsacks)
        ], axis=0)
        capacity_constraints = np.transpose(np.array([
            np.sum([
                item[0] * x[:, j * len(self.items) + i]
                for i, item in enumerate(self.items)
            ], axis=0) - capacity
            for j, capacity in enumerate(self.knapsacks)
        ]))
        number_of_items_constraint = np.transpose(np.array([
            np.sum([
                x[:, j * len(self.items) + i]
                for j, _ in enumerate(self.knapsacks)
            ], axis=0) - 1
            for i, _ in enumerate(self.items)
        ]))
        out["G"] = np.column_stack([capacity_constraints, number_of_items_constraint])

    def x_to_matrix(self, x):
        return x.reshape((len(self.knapsacks), len(self.items))).astype(int)
