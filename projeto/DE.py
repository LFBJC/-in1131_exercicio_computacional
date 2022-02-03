from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.factory import get_problem
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize

class ourDE(object):
    def __init__(self) -> None:
        pass

    def solver(problem=None, init_pop=100):
        algorithm = DE(
            pop_size=init_pop,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        )

        res = minimize(problem,
                    algorithm,
                    seed=1,
                    verbose=False)

        print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
