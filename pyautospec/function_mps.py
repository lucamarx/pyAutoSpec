"""
Mps based function compression algorithm
"""
import numpy as np
import itertools

from typing import List, Tuple

from .mps import Mps
from .plots import function_mps_comparison_chart, mps_entanglement_entropy_chart, function_mps_path_value_chart


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


    def _all_paths(self) -> np.ndarray:
        """
        Enumerate all paths in the model
        """
        return np.array([np.array(p) for p in itertools.product(*([range(A.shape[0]) for A in self.model[1:]]))])


    def _all_encodings(self) -> List[Tuple[np.ndarray, float]]:
        """
        Enumerate all possible argument encodings
        """
        return [(np.array(w), word2real(list(w), x0=self.x0, x1=self.x1)) for w in itertools.product(*([[0,1]] * len(self.model)))]


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


    def __mul__(self, b):
        """
        Compute the scalar product of the underlying models

        ╭───┐ ╭───┐ ╭───┐       ╭───┐
        │A_1├─┤A_2├─┤A_3├─ ... ─┤A_n│
        └─┬─┘ └─┬─┘ └─┬─┘       └─┬─┘
        ╭─┴─┐ ╭─┴─┐ ╭─┴─┐       ╭─┴─┐
        │B_1├─┤B_2├─┤B_3├─ ... ─┤B_n│
        └───┘ └───┘ └───┘       └───┘

        """
        return self.model.__mul__(b.model)


    def comparison_chart(self, n_points : int = 50, paths_threshold : float = None):
        """
        Compare the two functions

        Parameters:
        -----------

        n_points : int
        the number of points in the plot

        paths_threshold : float
        if it is not none displays a second plot with the number of the paths
        that contribute more than `paths_threshold` to the final value
        """
        function_mps_comparison_chart(self, n_points=n_points, paths_threshold=paths_threshold)


    def entanglement_entropy_chart(self):
        """
        Plot entanglement entropy chart
        """
        mps_entanglement_entropy_chart(self.model)


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


    def path_state_weight(self, path : np.ndarray, X : np.ndarray) -> float:
        """
        Evaluate contribution of an mps path when contracting with value X
        """
        N = len(self.model)

        if path.ndim != 1:
            raise Exception("path must be a one-dimensional array")

        if path.shape[0] != N-1:
            raise Exception("invalid path length, it must be {}".format(N-1))

        if X.ndim != 1:
            raise Exception("X must be a one-dimensional array of zeros and ones")

        if X.shape[0] != N:
            raise Exception("invalid X length, it must be {}".format(N))

        return self.model.path_state_weight(path, one_hot(N, self.model.part_d, [X])[0])


    def path_state_weights(self, sort : bool = False, error_threshold : float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get contributions to the final value by paths and function argument.

        If SORT is true sort paths by absolute contribution. If ERROR_THRESHOLD
        keep only the paths that keep the reconstruction error below error
        threshold.

        Returns:

        - the contributing paths
        - all the encodings
        - the weight matrix
        - the reconstruction error
        """
        all_paths, all_xencs = self._all_paths(), self._all_encodings()

        W = np.zeros((len(all_paths), len(all_xencs)))

        i = 0
        for path in all_paths:
            j = 0
            for x in all_xencs:
                W[i,j] = self.path_state_weight(path, x[0])
                j += 1
            i += 1

        f = np.sum(W, axis=0)
        if sort or error_threshold is not None:
            idxs = np.flip(np.argsort(np.max(np.abs(W), axis=1)))
            all_paths, W = all_paths[idxs], W[idxs,:]

            if error_threshold is not None:
                i, j = W.shape[0], 0
                while True:
                    assert(i > j)
                    if i-j < 2:
                        break
                    h = j + (i-j) // 2
                    e = np.average(np.abs(f - np.sum(W[0:h,:], axis=0)))
                    i,j = (i,h) if e > error_threshold else (h,j)

                all_paths, W = all_paths[0:i,:], W[0:i,:]

        err = np.average(np.abs(f - np.sum(W, axis=0)))

        return all_paths, all_xencs, W, err


    def path_value_chart(self, log : bool = False, sort : bool = False, error_threshold : float = None, threshold : float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Plot contributions to the final value by paths and function argument
        """
        return function_mps_path_value_chart(self, log=log, sort=sort, error_threshold=error_threshold, threshold=threshold)
