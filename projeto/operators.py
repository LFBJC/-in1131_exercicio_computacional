import numpy as np
from pymoo.core.sampling import Sampling


class SamplingRespectingPrecedence(Sampling):
    def __init__(self, pop_ratio=0.8, max_depth=30):
        super().__init__()
        self.pop_ratio = pop_ratio
        self.max_depth = max_depth

    def _do(self, problem, n_samples, **kwargs):
        modified_samples = int(self.pop_ratio*n_samples)
        x = np.random.rand(n_samples, problem.n_var)*(problem.xu - problem.xl) + problem.xl
        # there is no point in waiting for nothing until the first task starts
        x = (x-np.repeat(np.min(x, axis=1)[:, np.newaxis], problem.n_var, axis=1)+problem.xl)
        for sample in range(modified_samples):
            out = {}
            problem._evaluate(x[[sample], :], out)
            while max(out["G"][0]) > 0:
                x[sample, :] = np.random.rand(1, problem.n_var)*(problem.xu - problem.xl) + problem.xl
                # there is no point in waiting for nothing until the first task starts
                x = (x - np.repeat(np.min(x, axis=1)[:, np.newaxis], problem.n_var, axis=1) + problem.xl)
                out = {}
                problem._evaluate(x[[sample], :], out)
        return x


class SamplingWithSelection(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        x = np.random.rand(n_samples*300, problem.n_var) * (problem.xu - problem.xl) + problem.xl
        # there is no point in waiting for nothing until the first task starts
        x = (x - np.repeat(np.min(x, axis=1)[:, np.newaxis], problem.n_var, axis=1) + problem.xl)
        out = {}
        problem._evaluate(x, out)
        out["CV"] = np.sum(out["G"], axis=1)
        out["feasible"] = out["CV"] <= 0
        feasible = x[out["feasible"], :]
        infeasible = x[~out["feasible"], :]
        feasible = feasible[np.argsort(out["F"]), :]
        if feasible.shape[0] < n_samples:
            x = np.append(feasible, infeasible[np.argsort(out["CV"]+0.001*out["F"])])
        else:
            x = feasible
        return x