"""
Matrix product state class
"""
import numpy as np

from typing import List, Tuple

from .dmrg_learning import _cost, _log_likelihood_sample, _squared_norm, _fit_regression, _fit_classification


class Mps:
    """
    Matrix Product State

    ╭───┐ ╭───┐ ╭───┐       ╭───┐
    │ 1 ├─┤ 2 ├─┤ 3 ├─ ... ─┤ N │
    └─┬─┘ └─┬─┘ └─┬─┘       └─┬─┘
      │     │     │           │
    """

    def __init__(self, N : int, part_d : int = 2, max_bond_d : int = 20):
        """
        Initialize a random matrix product state

        Parameters:
        -----------

        N : int
        number of particles

        part_d : int
        particle dimension

        max_bond_d: int
        maximum bond dimension
        """
        if N < 3:
            raise Exception("chain too short")

        if part_d < 2:
            raise Exception("particle dimension must be at least 2")

        self.N = N
        self.part_d = part_d
        self.max_bond_d = max_bond_d

        # start with small bond dimension
        bond_d = 2

        # setup tensor container
        self.mps = [np.random.rand(*s) for s in [(part_d,bond_d)] + [(bond_d,part_d,bond_d)]*(N-2) + [(bond_d,part_d)]]

        # setup contraction cache
        self.cache = [None]*self.N

        # make mps left canonical
        for n in range(self.N-1):
            if n > 0:
                m = self[n].reshape((self.part_d*bond_d, bond_d))
            else:
                m = self[n]

            u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

            if n > 0:
                self[n] = u.reshape((bond_d, self.part_d, bond_d))
            else:
                self[n] = u

            if n < N-2:
                self[n+1] = np.einsum("i,ij,jqk->iqk", s, v, self[n+1])
            else:
                self[n+1] = np.einsum("i,ij,jq->iq", s, v, self[n+1])


    def __len__(self) -> int:
        """
        The number of matrices in the chain
        """
        return self.N


    def __getitem__(self, n : int) -> np.ndarray:
        """
        Get matrix at site n
        """
        return self.mps[n]


    def __setitem__(self, n : int, m : np.ndarray):
        """
        Set matrix at site n truncating exceeding bond dimensions
        """
        if n == 0:
            self.mps[n] = m[:, 0:self.max_bond_d]

        elif n == self.N-1:
            self.mps[n] = m[0:self.max_bond_d, :]

        else:
            self.mps[n] = m[0:self.max_bond_d, :, 0:self.max_bond_d]


    def bond_dimension(self) -> Tuple[int, int]:
        """
        Return the bond dimension

        Returns:
        --------

        A tuple (final bond dimension, max bond dimension)
        """
        return self[self.N // 2].shape[2], self.max_bond_d


    def __call__(self, X : np.ndarray) -> np.ndarray:
        """
        Evaluate MPS on batch X[b,n,p]

        ╭───┐ ╭───┐       ╭───┐
        │ 1 ├─┤ 2 ├─ ... ─┤ N │
        └─┬─┘ └─┬─┘       └─┬─┘
          ◯     ◯           ◯

        Parameters:
        -----------

        X : np.ndarray
        a batch of N part_d dimensional vectors

        Returns:
        --------

        the value of the tensor for the batch X
        """
        if len(X.shape) == 2:
            X = X.reshape((1, *X.shape))

        if X.shape[1] < self.N:
            raise Exception("X is too short")

        T = np.einsum("bp,pj->bj", X[:,0,:], self[0])
        for n in range(1,self.N-1):
            T = np.einsum("bi,bp,ipj->bj", T, X[:,n,:], self[n])

        T = np.einsum("bi,bp,ip->b", T, X[:,self.N-1,:], self[self.N-1])

        return T


class MpsR(Mps):
    """
    Mps for regression
    """

    def __repr__(self) -> str:
        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

  particle dim: {:3d}
      bond dim: {:3d} (max: {:d})
        """.format(self.N, self.part_d, *self.bond_dimension())


    def cost(self, X : np.ndarray, y : np.ndarray) -> Tuple[float, float, float]:
        """
        Compute cost function
        """
        return _cost(self, X, y)


    def fit(self, X : np.ndarray, y : np.ndarray, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
        """
        Fit the MPS to the data

        0. for each epoch
        1.  sample a random mini-batch from X
        2.  sweep right → left (left → right)
        3.   contract A^k and A^(k+1) into B^k
        4.   evaluate gradients for mini-batch
        5.   update B^k
        6.   split B^k with SVD ensuring canonicalization
        7.   move to next k

        Parameters:
        -----------
        X : np.ndarray

        y : np.ndarray
        the data to be fitted

        learn_rate : float
        learning rate

        batch_size : int
        batch size

        epochs : int
        number of epochs
        """
        _fit_regression(self, X, y, learn_rate, batch_size, epochs)

        return self


class MpsC(Mps):
    """
    Mps for classification
    """

    def __init__(self, N : int, part_d : int = 2, max_bond_d : int = 20):
        super().__init__(N, part_d, max_bond_d)

        # normalize
        t = self[self.N-1]
        norm = np.sqrt(np.einsum("pi,pi->", t, t))
        self[self.N-1] = t / norm


    def __repr__(self) -> str:
        norm = np.sqrt(np.einsum("pi,pi->", self[self.N-1], self[self.N-1]))

        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

          norm: {:.2f}
  particle dim: {:3d}
      bond dim: {:3d} (max: {:d})
        """.format(self.N, norm, self.part_d, *self.bond_dimension())


    def squared_norm(self) -> float:
        """
        Compute squared norm
        """
        return _squared_norm(self)


    def log_likelihood(self, X : np.ndarray) -> np.ndarray:
        """
        Compute cost function
        """
        return _log_likelihood_sample(self, X)


    def fit(self, X : np.ndarray, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
        """
        Fit the MPS to the data

        0. for each epoch
        1.  sample a random mini-batch from X
        2.  sweep right → left (left → right)
        3.   contract A^k and A^(k+1) into B^k
        4.   evaluate gradients for mini-batch
        5.   update B^k
        6.   split B^k with SVD ensuring canonicalization
        7.   move to next k

        Parameters:
        -----------
        X : np.ndarray

        learn_rate : float
        learning rate

        batch_size : int
        batch size

        epochs : int
        number of epochs
        """
        _fit_classification(self, X, learn_rate, batch_size, epochs)

        return self


class SymbolicMps():
    """
    A symbolic matrix product state operate on fixed length sequences of
    symbols.

    These are converted into vectors by one-hot encoding.
    """

    def __init__(self, N : int, alphabet : int = 2, max_bond_d : int = 20):
        """
        Initialize a random symbolic matrix product state

        Parameters:
        -----------

        N : int
        sequence length

        alphabet : int
        alphabet size

        max_bond_d: int
        maximum bond dimension
        """
        self.mps = MpsR(N, alphabet, max_bond_d)


    def __repr__(self) -> str:
        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

  alphabet size: {:3d}
       bond dim: {:3d} (max: {})
        """.format(len(self.mps), self.mps.part_d, *self.mps.bond_dimension())


    def __len__(self) -> int:
        """
        The number of matrices in the chain
        """
        return len(self.mps)


    def _one_hot(self, X : List[List[int]]) -> np.ndarray:
        """
        Perform one-hot encoding
        """
        idxs = np.array(X).reshape(-1)
        return np.eye(self.mps.part_d)[idxs].reshape((-1, self.mps.N, self.mps.part_d))


    def __call__(self, X : List[List[int]]) -> np.ndarray:
        """
        Evaluate MPS on words

        ╭───┐ ╭───┐       ╭───┐
        │ 1 ├─┤ 2 ├─ ... ─┤ N │
        └─┬─┘ └─┬─┘       └─┬─┘
          ◯     ◯           ◯

        Parameters:
        -----------

        X : List[List[int]]
        a list ow words in the mps alphabet

        Returns:
        --------

        the values of the tensor
        """
        return self.mps(self._one_hot(X))


    def bond_dimension(self) -> Tuple[int, int]:
        """
        Return the bond dimension

        Returns:
        --------

        A tuple (final bond dimension, max bond dimension)
        """
        return self.mps.bond_dimension()


    def fit(self, X : List[List[int]], y : List[float], learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
        """
        Fit the MPS to the data
        """
        self.mps.fit(self._one_hot(X), np.array(y), learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)
        return self
