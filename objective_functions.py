import numpy as np
from pymoo.core.problem import Problem, ElementwiseProblem

#1.a
class Ackley(Problem):
    def __init__(self):
        self.name = 'Ackley'
        super().__init__(n_var=2,
                         n_obj=1,
                         n_constr=0,
                         xl=-32.768,
                         xu=32.768)

    def _evaluate(self, x, out, *args, **kwargs):
        part1 = -20 * np.exp(-0.2 * np.sqrt((1/self.n_var)*np.sum(x ** 2, axis=1)))
        part2 = - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e
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
        part2 = - np.prod([np.cos(x[:, i]/np.sqrt(i+1)) for i in range(x.shape[1])])
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

