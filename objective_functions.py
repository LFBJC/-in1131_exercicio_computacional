import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem
import autograd.numpy as anp

#1.a
class Ackley(Problem):
    def __init__(self):
        self.name = 'Ackley'
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=-32.768,
                         xu=32.768)
        self.a=20
        self.b=0.2
        self.c=2

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = -20 * np.exp(-0.2 * np.sqrt((1/self.n_var)*np.sum(x ** 2, axis=1)))
        part2 = - np.exp(np.mean(np.cos(2 * np.pi * x), axis=1)) + 20 + np.e
        out["F"] = part1 + part2


#1.b
class Griewank(Problem):
    def __init__(self):
        self.name = 'Griewank'
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=-600,
                         xu=600)

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = np.sum(x**2, axis=1)/4000
        part2 = - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[1] + 1))), axis=1)
        out["F"] = part1 + part2 + 1

class Colville(Problem):
    def __init__(self):
        self.name = 'Colville'
        super().__init__(n_var=4,
                         n_obj=1,
                         n_constr=0,
                         xl=-10,
                         xu=10)

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = 100*(x[:, 0]**2 - x[:, 1])**2 + (x[:, 0] - np.ones_like(x[:, 0]))**2
        part2 = (x[:, 2] - np.ones_like(x[:, 2]))**2 + 90*(x[:, 2]**2 - x[:, 3])**2
        part3 = 10.1*((x[:, 1] - np.ones_like(x[:, 1]))**2 + (x[:, 3] - np.ones_like(x[:, 3]))**2)
        part4 = 19.8*(x[:, 1] - np.ones_like(x[:, 1]))*(x[:, 3] - np.ones_like(x[:, 3]))
        out["F"] = part1 + part2 + part3 + part4


#1.c 
class Trid(ElementwiseProblem):
    def __init__(self):
        self.name = 'Trid'
        super().__init__(n_var=5,
                         n_obj=1,
                         n_constr=0,
                         xl=-20,
                         xu=20)

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = np.sum((x - 1) ** 2) 
        part2 = - np.sum(x[1:] * x[:-1])
        out["F"] = part1 + part2

#2
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
