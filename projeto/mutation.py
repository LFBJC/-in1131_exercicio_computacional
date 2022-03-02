
import re

from pymoo.config import Config
from pymoo.problems.many import *
from pymoo.problems.multi import *
from pymoo.problems.single import *


import numpy as np

from pymoo.core.mutation import Mutation
from pymoo.operators.repair.to_bound import set_to_bounds_if_outside_by_problem

class B_Flip_Mutation(Mutation):

    def __init__(self, prob=None):
        super().__init__()
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        X = X.astype(np.bool)
        _X = np.full(X.shape, np.inf)

        M = np.random.random(X.shape)
        flip, no_flip = M < self.prob, M >= self.prob

        _X[flip] = np.logical_not(X[flip])
        _X[no_flip] = X[no_flip]

        return _X.astype(np.bool)

class PolMut(Mutation):
    def __init__(self, eta, prob=None):
        super().__init__()
        self.eta = float(eta)

        if prob is not None:
            self.prob = float(prob)
        else:
            self.prob = None

    def _do(self, problem, X, **kwargs):

        X = X.astype(float)
        Y = np.full(X.shape, np.inf)

        if self.prob is None:
            self.prob = 1.0 / problem.n_var

        do_mutation = np.random.random(X.shape) < self.prob

        Y[:, :] = X

        xl = np.repeat(problem.xl[None, :], X.shape[0], axis=0)[do_mutation]
        xu = np.repeat(problem.xu[None, :], X.shape[0], axis=0)[do_mutation]

        X = X[do_mutation]

        delta1 = (X - xl) / (xu - xl)
        delta2 = (xu - X) / (xu - xl)

        mut_pow = 1.0 / (self.eta + 1.0)

        rand = np.random.random(X.shape)
        mask = rand <= 0.5
        mask_not = np.logical_not(mask)

        deltaq = np.zeros(X.shape)

        xy = 1.0 - delta1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (self.eta + 1.0)))
        d = np.power(val, mut_pow) - 1.0
        deltaq[mask] = d[mask]

        xy = 1.0 - delta2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (self.eta + 1.0)))
        d = 1.0 - (np.power(val, mut_pow))
        deltaq[mask_not] = d[mask_not]

        # mutated values
        _Y = X + deltaq * (xu - xl)

        # back in bounds if necessary (floating point issues)
        _Y[_Y < xl] = xl[_Y < xl]
        _Y[_Y > xu] = xu[_Y > xu]

        # set the values for output
        Y[do_mutation] = _Y

        # in case out of bounds repair (very unlikely)
        Y = set_to_bounds_if_outside_by_problem(problem, Y)

        return Y

# =========================================================================================================
# Generic
# =========================================================================================================


def get_from_list(l, name, args, kwargs):
    i = None

    for k, e in enumerate(l):
        if e[0] == name:
            i = k
            break

    if i is None:
        for k, e in enumerate(l):
            if re.match(e[0], name):
                i = k
                break

    if i is not None:

        if len(l[i]) == 2:
            name, clazz = l[i]

        elif len(l[i]) == 3:
            name, clazz, default_kwargs = l[i]

            # overwrite the default if provided
            for key, val in kwargs.items():
                default_kwargs[key] = val
            kwargs = default_kwargs

        return clazz(*args, **kwargs)
    else:
        raise Exception("Object '%s' for not found in %s" % (name, [e[0] for e in l]))

# =========================================================================================================
# Mutation
# =========================================================================================================

def get_mutation_options():
    from pymoo.operators.mutation.nom import NoMutation
    #from pymoo.operators.mutation.bitflip import BinaryBitflipMutation
    #from pymoo.operators.mutation.pm import PolynomialMutation
    from pymoo.operators.integer_from_float_operator import IntegerFromFloatMutation
    from pymoo.operators.mutation.inversion import InversionMutation

    MUTATION = [
        ("none", NoMutation, {}),
        ("pol_mut", PolMut, dict(eta=20)),
        #("int_pm", IntegerFromFloatMutation, dict(clazz=PolynomialMutation, eta=20)),
        ("bitflip", B_Flip_Mutation),
        ("perm_inv", InversionMutation)
    ]

    return MUTATION


def get_mutation(name, *args, d={}, **kwargs):
    return get_from_list(get_mutation_options(), name, args, {**d, **kwargs})

