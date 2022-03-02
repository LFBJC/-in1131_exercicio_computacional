import numpy as np
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.operators.crossover.ox import random_sequence
from pymoo.operators.mutation.inversion import inversion_mutation


def sigmoid(x):
    return 1/(1+np.exp(-x))


class OurInversionMutation(Mutation):
    def __init__(self, prob=1.0):
        """
        This mutation is applied to permutations. It randomly selects a segment of a chromosome and reverse its order.
        For instance, for the permutation `[1, 2, 3, 4, 5]` the segment can be `[2, 3, 4]` which results in `[1, 4, 3, 2, 5]`.
        Parameters
        ----------
        prob : float
            Probability to apply the mutation to the individual

        """
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, y in enumerate(X):
            if np.random.random() < self.prob*(1-sigmoid(np.mean(np.std(X, axis=0)))):
                seq = random_sequence(len(y))
                Y[i] = inversion_mutation(y, seq, inplace=True)
        return Y


class OurInversionMutation2(Mutation):
    def __init__(self, prob=1.0):
        """
        This mutation is applied to permutations. It randomly selects a segment of a chromosome and reverse its order.
        For instance, for the permutation `[1, 2, 3, 4, 5]` the segment can be `[2, 3, 4]` which results in `[1, 4, 3, 2, 5]`.
        Parameters
        ----------
        prob : float
            Probability to apply the mutation to the individual

        """
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, y in enumerate(X):
            seq = random_sequence(len(y))
            if np.random.random() < self.prob*(1-sigmoid(np.mean(np.std(X[:, seq[0]:seq[1]], axis=0)))):
                Y[i] = inversion_mutation(y, seq, inplace=True)
        return Y


class OurMutation2(Mutation):
    def __init__(self, prob=1.0):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        Y = X.copy()
        for i, y in enumerate(X):
            for j in range(len(y)):
                if np.random.random() < self.prob*(1-sigmoid(np.std(X[j], axis=0))):
                    Y[i, j] = 1-y[j]
        return Y


class SamplingRespectingPrecedence(Sampling):
    def __init__(self, pop_ratio=0.8, max_depth=30):
        super().__init__()
        self.pop_ratio = pop_ratio
        self.max_depth = max_depth

    def _do(self, problem, n_samples, **kwargs):
        modified_samples = int(self.pop_ratio*n_samples)
        x = np.random.rand(n_samples*3, problem.n_var)*(problem.xu - problem.xl) + problem.xl
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
        depth = kwargs.get('depth') or 100000
        x_size = kwargs.get('x_size') or n_samples
        x = np.random.rand(x_size*3, problem.n_var) * (problem.xu - problem.xl) + problem.xl
        # there is no point in waiting for nothing until the first task starts
        x = (x - np.repeat(np.min(x, axis=1)[:, np.newaxis], problem.n_var, axis=1) + problem.xl)
        out = {}
        problem._evaluate(x, out)
        out["CV"] = np.max(out["G"], axis=1)
        out["feasible"] = out["CV"] <= 0
        feasible = x[out["feasible"], :]
        infeasible = x[~out["feasible"], :]
        feasible = feasible[np.argsort(out["F"][out["feasible"]]), :]
        infeasibility_values = out["CV"][~out["feasible"]] + 0.001 * out["F"][~out["feasible"]]
        infeasible = infeasible[np.argsort(infeasibility_values)]
        while feasible.shape[0] < n_samples and depth > 0:
            x = np.random.rand(x_size * 3, problem.n_var) * (problem.xu - problem.xl) + problem.xl
            # there is no point in waiting for nothing until the first task starts
            x = (x - np.repeat(np.min(x, axis=1)[:, np.newaxis], problem.n_var, axis=1) + problem.xl)
            out = {}
            problem._evaluate(x, out)
            out["CV"] = np.max(out["G"], axis=1)
            out["feasible"] = out["CV"] <= 0
            inner_feasible = x[out["feasible"], :]
            inner_infeasible = x[~out["feasible"], :]
            inner_feasible = inner_feasible[np.argsort(out["F"][out["feasible"]]), :]
            inner_infeasibility_values = out["CV"][~out["feasible"]] + 0.001 * out["F"][~out["feasible"]]
            infeasible = np.append(infeasible, inner_infeasible, axis=0)
            infeasible = infeasible[np.argsort(list(infeasibility_values)+list(inner_infeasibility_values))]
            infeasible = infeasible[:(n_samples - feasible.shape[0])]
            infeasibility_values = np.sort(list(infeasibility_values)+list(inner_infeasibility_values))
            infeasibility_values = infeasibility_values[:(n_samples - feasible.shape[0])]
            feasible = np.append(feasible, inner_feasible, axis=0)
            depth -= 1
        if feasible.shape[0] < n_samples:
            x = np.append(feasible, infeasible, axis=0)
        elif feasible.shape[0] > n_samples:
            x = feasible[:n_samples, :]
        else:
            x = feasible
        return x