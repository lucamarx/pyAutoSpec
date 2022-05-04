"""
Matrix product state
"""

import numpy as np


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

        # setum tensor container
        self.mps = [np.random.rand(*s) for s in [(part_d,bond_d)] + [(bond_d,part_d,bond_d)]*(N-2) + [(bond_d,part_d)]]
        self.cache = None

        # make mps left canonical
        for n in range(self.N-1):
            if n > 0:
                m = self._get(n).reshape((self.part_d*bond_d, bond_d))
            else:
                m = self._get(n)

            u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

            if n > 0:
                self._set(n, u.reshape((bond_d, self.part_d, bond_d)))
            else:
                self._set(n, u)

            if n < N-2:
                self._set(n+1, np.einsum("i,ij,jqk->iqk", s, v, self._get(n+1)))
            else:
                self._set(n+1, np.einsum("i,ij,jq->iq", s, v, self._get(n+1)))

        # normalize
        t = self._get(self.N-1)
        norm = np.sqrt(np.einsum("pi,pi->", t, t))
        self._set(self.N-1, t / norm)


    def _get(self, n : int) -> np.ndarray:
        """
        Get matrix at site n
        """
        return self.mps[n]


    def _set(self, n : int, m : np.ndarray):
        """
        Set matrix at site n truncating exceeding bond dimensions
        """
        if n == 0:
            self.mps[n] = m[:, 0:self.max_bond_d]

        elif n == self.N-1:
            self.mps[n] = m[0:self.max_bond_d, :]

        else:
            self.mps[n] = m[0:self.max_bond_d, :, 0:self.max_bond_d]


    def __repr__(self) -> str:
        norm = np.sqrt(np.einsum("pi,pi->", self._get(self.N-1), self._get(self.N-1)))

        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

          norm:  {:.2f}
  particle dim:  {}
  max bond dim:  {}
        """.format(self.N, norm, self.part_d, self.max_bond_d)


    def __call__(self, X : np.ndarray) -> float:
        """
        Evaluate MPS on batch X[b,n,p]

        ╭───┐ ╭───┐       ╭───┐
        │ 1 ├─┤ 2 ├─ ... ─┤ N │
        └─┬─┘ └─┬─┘       └─┬─┘
          ◯     ◯           ◯
        """
        if len(X.shape) == 2:
            X = X.reshape((1, *X.shape))

        if X.shape[1] < self.N:
            raise Exception("X is too short")

        T = np.einsum("bp,pj->bj", X[:,0,:], self._get(0))
        for n in range(1,self.N-1):
            T = np.einsum("bi,bp,ipj->bj", T, X[:,n,:], self._get(n))

        T = np.einsum("bi,bp,ip->b", T, X[:,self.N-1,:], self._get(self.N-1))

        return T
