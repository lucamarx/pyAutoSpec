"""
Mps based function compression algorithm
"""
import numpy as np
import itertools
import matplotlib.pyplot as plt

from typing import List
from .mps import SymbolicMps


def word2real(s : List[int], x0 : float = 0.0, x1 : float = 1.0) -> float:
    """
    Convert the binary representation s of xϵ[x0,x1) into the number itself
    """
    s = [0] + s
    return x0 + sum([s[i] * 2**(-i) for i in range(len(s))]) * (x1-x0)


def real2word(r : float, l : int = 8, x0 : float = 0.0, x1 : float = 1.0) -> List[int]:
    """
    Convert a real number xϵ[x0,x1) into its binary representation (with
    maximum length l)
    """
    if r < x0 or r >= x1:
        raise Exception("out of bounds")

    r = (r - x0) / (x1 - x0)
    w = []
    for _ in range(0,l+1):
        d = 1 if r >= 1 else 0
        w.append(d)
        r = (r-d)*2
    return w[1:]


class FunctionMps():
    """
    Mps based real function model
    """

    def __init__(self, f, x0 : float = 0.0, x1 : float = 1.0, sequence_length : int = 8, max_bond_dim : int = 20):
        """
        Intialize a model of a real function f: [x0,x1) → R
        """
        self.f = f

        self.x0 = x0
        self.x1 = x1

        self.model = SymbolicMps(sequence_length, 2, max_bond_dim)

        data = [(list(x), f(word2real(x, x0=x0, x1=x1))) for x in itertools.product(*([[0,1]] * sequence_length))]

        self.model.fit([t[0] for t in data], [t[1] for t in data])


    def __repr__(self):
        return "MPS(N={}) {}: [{:.2f},{:.2f}] → R".format(self.model.N, self.f.__repr__(), self.x0, self.x1)


    def __call__(self, X : float) -> float:
        """
        Parameters:
        -----------

        X : float
        a point in [x0,x1)

        Returns:
        --------

        the value of the function at x
        """
        return self.model([real2word(X, l=self.model.N)])[0]
    def comparison_chart(self, n_points : int = 50):
        """
        Compare the two functions
        """
        xs = np.linspace(self.x0, self.x1, endpoint = False, num = n_points)

        v0 = np.array([self.f(x) for x in xs])
        v1 = np.array([self(x) for x in xs])

        error = np.abs(v1 - v0)

        plt.figure()

        plt.title("{} reconstruction error: avg={:.2f} max={:.2f} ".format(self.f.__repr__(), np.average(error), np.max(error)))

        plt.plot(xs, v0, label="original f")
        plt.plot(xs, v1, label="f")

        plt.legend()
