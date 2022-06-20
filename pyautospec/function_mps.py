"""
Mps based function compression algorithm
"""
import numpy as np
import itertools

from typing import List, Tuple

from .mps import Mps
from .plots import function_wfa_comparison_chart


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


def one_hot(N : int, part_d : int, X : List[List[int]]) -> np.ndarray:
    """
    Perform one-hot encoding
    """
    idxs = np.array(X).reshape(-1)
    return np.eye(part_d)[idxs].reshape((-1, N, part_d))


class FunctionMps():
    """
    Mps based real function model
    """

    def __init__(self, sequence_length : int = 8, max_bond_dim : int = 20):
        """
        Intialize a model of a real function f: [x0,x1) → R

        Parameters:
        -----------

        sequence_length : int
        the underlying MPS length

        max_bond_dim : int
        the underlying MPS maximum bond dimension
        """
        self.f, self.x0, self.x1 = None, None, None

        self.model = Mps(sequence_length, 2, max_bond_dim)


    def __repr__(self) -> str:
        if self.f is None:
            return "  FunctionMps(N={}) <?>: [<?>,<?>] → R\n{}".format(len(self.model), self.model.__repr__())
        else:
            return "  FunctionMps(N={}) {}: [{:.2f},{:.2f}] → R\n{}".format(len(self.model), self.f.__repr__(), self.x0, self.x1, self.model.__repr__())


    def _encode(self, x : float):
        """
        Encode real value into vector word
        """
        return one_hot(len(self.model), self.model.part_d, [real2word(x, l=len(self.model), x0=self.x0, x1=self.x1)])


    def __call__(self, x : float) -> float:
        """
        Evaluate learned function at x

        Parameters:
        -----------

        x : float
        a point in [x0,x1)

        Returns:
        --------

        the value of the function at x
        """
        return self.model(self._encode(x))[0]


    def comparison_chart(self, n_points : int = 50):
        """
        Compare the two functions

        Parameters:
        -----------

        n_points : int
        the number of points in the plot
        """
        function_wfa_comparison_chart(self, n_points, None, plot_derivative = False)


    def fit(self, f, x0 : float = 0.0, x1 : float = 1.0, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
        """
        Fit the model to the function f defined on the interval [x0,x1)

        Parameters:
        -----------

        f : function
        the function to be fitted

        x0 : float
        x1 : float
        the interval the function is defined on

        learn_rate : float
        the learning rate

        batch_size : int
        the batch size used at each step

        epochs : int
        the number of epochs

        Returns:
        --------

        The object itself
        """
        self.f = f
        self.x0 = x0
        self.x1 = x1

        data = [(list(x), f(word2real(list(x), x0=x0, x1=x1))) for x in itertools.product(*([[0,1]] * len(self.model)))]

        self.model.fit(one_hot(len(self.model), self.model.part_d, np.array([t[0] for t in data])), np.array([t[1] for t in data]), learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

        return self


    def paths_weights(self, x : float, threshold : float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enumerate all paths contributing to the final value
        """
        return self.model.paths_weights(self._encode(x)[0,:,:], threshold=threshold)
