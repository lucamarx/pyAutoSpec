"""
Matrix product state class
"""
import itertools

import numpy as np
import jax.numpy as jnp

from jax import jit, vmap
from typing import List, Tuple

from .plots import training_chart
from .dmrg_learning import cost, fit_regression


@jit
def path_weight(A : List[jnp.ndarray], path : jnp.ndarray) -> float:
    """
    Compute the weight of a path in the state graph
    """
    weight = A[0][path[0]]

    for i in range(1, path.shape[0]):
        p, q = path[i-1], path[i]
        weight *= A[i][p, q]

    weight *= A[-1][path[-1]]

    return weight


def contributing_paths(paths : np.ndarray, weights : np.ndarray, threshold : float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find paths that provide net contributions to final weight. If a
    threshold is specified discard the lowest weights
    """
    # sort weights
    sort_idx = np.argsort(weights)
    wgt = weights[sort_idx]

    # separate positive and negative weights
    p, n = wgt[wgt>0], np.flip(-wgt[wgt<0])

    zero_idx = np.argmin(wgt < 0)

    i = j = 0
    # find 1-to-1 matches of a positive and a negative weights and cancel them
    while i < p.shape[0] and j < n.shape[0]:

        assert(p[i] == weights[sort_idx[zero_idx+i]])
        assert(n[j] == -weights[sort_idx[zero_idx-j-1]])

        if abs(p[i] - n[j]) < 1e-8:
            p[i] = n[j] = 0

            # cancel the original unsorted weights
            weights[sort_idx[zero_idx+i]] = weights[sort_idx[zero_idx-j-1]] = 0

            i += 1
            j += 1

            continue

        # bring up another match
        if p[i] < n[j]:
            i += 1
        else:
            j += 1

    # discard lowest weights
    if threshold is not None:
        cp, cn = np.cumsum(p), np.cumsum(n)
        ci, cj = np.argmin(cp < np.sum(p)*threshold), np.argmin(cn < np.sum(n)*threshold)

        weights[sort_idx[zero_idx:zero_idx+ci]] = 0
        weights[sort_idx[zero_idx-cj-1:zero_idx-1]] = 0

    nz_idx = np.nonzero(weights)

    return paths[nz_idx], weights[nz_idx]


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

        # initialize singular values cache
        self.singular_values = [None] * (N-1)

        # initialize training/validation costs
        self.train_costs, self.valid_costs = [], []


    def __repr__(self) -> str:
        bond_d = max([self[n].shape[-1] for n in range(len(self)-1)])

        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

  particle dim: {:3d}
      bond dim: {:3d} (max: {:d})
        """.format(self.N, self.part_d, bond_d, self.max_bond_d)


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


    def bond_dimensions(self) -> List[int]:
        """
        Return the bond dimensions

        Returns:
        --------

        A list of bond dimensions
        """
        return [self[n].shape[-1] for n in range(len(self)-1)]


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
        if X.ndim == 2:
            X = X.reshape((1, *X.shape))

        if X.shape[1] < self.N:
            raise Exception("X is too short")

        if X.shape[2] != self.part_d:
            raise Exception("invalid particle dimension")

        T = np.einsum("bp,pj->bj", X[:,0,:], self[0])
        for n in range(1,self.N-1):
            T = np.einsum("bi,bp,ipj->bj", T, X[:,n,:], self[n])

        T = np.einsum("bi,bp,ip->b", T, X[:,self.N-1,:], self[self.N-1])

        return T


    def __mul__(self, b):
        """
        Compute the scalar product with Mps b

        ╭───┐ ╭───┐ ╭───┐       ╭───┐
        │A_1├─┤A_2├─┤A_3├─ ... ─┤A_n│
        └─┬─┘ └─┬─┘ └─┬─┘       └─┬─┘
        ╭─┴─┐ ╭─┴─┐ ╭─┴─┐       ╭─┴─┐
        │B_1├─┤B_2├─┤B_3├─ ... ─┤B_n│
        └───┘ └───┘ └───┘       └───┘
        """
        a = self

        if a.part_d != b.part_d:
            raise Exception("arguments have different particle dimensions")

        if len(a) != len(b):
            raise Exception("arguments have different lengths")

        s = np.einsum("pj,pk->jk", a.mps[0], b.mps[0])

        for n in range(1,len(a)-1):
            s = np.einsum("jk,jpl,kpm->lm", s, a.mps[n], b.mps[n])

        s = np.einsum("jk,jp,kp->", s, a.mps[-1], b.mps[-1])

        return s


    def predict(self, X : np.ndarray) -> np.ndarray:
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
        return self(X)


    def cost(self, X : np.ndarray, y : np.ndarray) -> Tuple[float, float, float]:
        """
        Compute cost function
        """
        return cost(self, X, y)


    def fit(self, X_train : np.ndarray, y_train : np.ndarray, X_valid : np.ndarray = None, y_valid : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10, early_stop : bool = False):
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
        X_train : np.ndarray
        y_train : np.ndarray
        the training dataset

        X_valid : np.ndarray
        y_valid : np.ndarray
        the optional validation dataset

        learn_rate : float
        learning rate

        batch_size : int
        batch size

        epochs : int
        number of epochs

        early_stop : bool
        stop as soon as overfitting is detected (needs a validation dataset)
        """
        self.train_costs, self.valid_costs = fit_regression(self, X_train, y_train, X_valid, y_valid, learn_rate, batch_size, epochs, early_stop)

        return self


    def training_chart(self):
        """
        Plots training/validation costs
        """
        if len(self.train_costs) == 0:
            raise Exception("the model has not been trained yet")

        training_chart(self.train_costs, self.valid_costs)


    def paths_weights(self, X : np.ndarray, threshold : float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enumerate all paths contributing to the final value. If threshold is
        specified discard paths that contribute for less than threshold% to the
        total weight
        """
        if X.ndim != 2:
            raise Exception("X must contain a single v-word")

        if X.shape[0] < self.N:
            raise Exception("X is too short")

        if X.shape[1] != self.part_d:
            raise Exception("invalid particle dimension")

        if threshold is not None and (threshold <= 0 or threshold >= 1):
            raise Exception("invalid threshold, it must be > 0 and < 1")

        A = []
        # contract MPS with v-word
        A.append(np.einsum("pj,p->j", self[0], X[0,:]))
        for i in range(1, len(self)-1):
            A.append(np.einsum("ipj,p->ij", self[i], X[i,:]))
        A.append(np.einsum("ip,p->i", self[-1], X[-1,:]))

        # enumerate all paths
        paths = jnp.array(list(itertools.product(*([range(A[i].shape[0]) for i in range(1,len(A))]))), dtype=jnp.int32)

        # compute individual weights
        weights = vmap(lambda p: path_weight(A, p), in_axes=0)(paths)

        if threshold is not None:
            return contributing_paths(np.array(paths), np.array(weights.copy()), threshold)
        else:
            return np.array(paths), np.array(weights)


    def entanglement_entropy(self, n : int = None) -> float:
        """
        Compute the entanglement entropy between the first n and the
        remaining (N-n) particles
        """
        if n is None:
            sv2s = [np.square(sv / np.sum(sv)) for sv in self.singular_values]
            return [-np.einsum("i,i->", sv2, np.log2(sv2)).item() for sv2 in sv2s]
        else:
            sv2 = np.square(self.singular_values[n] / np.sum(self.singular_values[n]))
            return -np.einsum("i,i->", sv2, np.log2(sv2)).item()
