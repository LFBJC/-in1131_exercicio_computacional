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
        part1 = -20 * np.exp(-0.2 * np.sqrt((1/self.n_var)*np.sum(x ** 2, axis=1)))
        part2 = - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
        out["F"] = part1 + part2


class Griewank(Problem):
    def __init__(self):
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=-600,
                         xu=600)

    def _evaluate(self, x, out, *args, **kwargs):
        f1 = np.sum(x**2, axis=1)/4000 - np.prod(np.cos(x/np.sqrt(1j)), axis=1) + 1
        out["F"] = f1


class KnapSack(Problem):
    def __init__(self, knapsacks, items):
        self.items = items
        self.knapsacks = knapsacks
        super().__init__(n_var=len(knapsacks) * len(items),
                         n_obj=1,
                         n_constr=len(knapsacks),
                         xl=0,
                         xu=1,
                         type_var=np.int32)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum([
            item[1]*x[j*len(self.items)+i]
            for i, item in enumerate(self.items)
            for j, _ in enumerate(self.knapsacks)
        ], axis=1)
        out["G"] = [
            np.sum([
                item[0] * x[j * len(self.items) + i]
                for i, item in enumerate(self.items)
            ], axis=1) - capacity
            for j, capacity in enumerate(self.knapsacks)
        ]

