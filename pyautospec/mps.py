"""
Matrix product state class
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


    def _merge(self, k : int) -> np.ndarray:
        """
        Contract two adjacent rank 3 tensors into one rank 4 tensor

         ╭───┐ ╭───┐     ╭────┐
        ─┤ k ├─┤k+1├─ = ─┤    ├─
         └─┬─┘ └─┬─┘     └┬──┬┘

        or at the head/tail

         ╭───┐ ╭───┐     ╭────┐
         │ 0 ├─┤ 1 ├─ =  │    ├─
         └─┬─┘ └─┬─┘     └┬──┬┘

         ╭───┐ ╭───┐     ╭────┐
        ─┤N-1├─┤ N │  = ─┤    │
         └─┬─┘ └─┬─┘     └┬──┬┘
        """
        if 0 < k and k+1 < self.N-1:
            return np.einsum("ipj,jqk->ipqk", self._get(k), self._get(k+1))

        if k == 0:
            return np.einsum("pj,jqk->pqk", self._get(0), self._get(1))

        if k+1 == self.N-1:
            return np.einsum("ipj,jq->ipq", self._get(self.N-2), self._get(self.N-1))

        raise Exception("invalid k")


    def _split(self, k : int, B : np.ndarray, left : bool = True) -> np.ndarray:
        """
        Split a rank 3/4 tensor into two adjacent left/right canonical rank 2/3 tensors

          ╭────┐     ╭───┐   ╭───┐
         ─┤    ├─ = ─┤   ├─ ─┤   ├─
          └┬──┬┘     └─┬─┘   └─┬─┘
        """
        if len(B.shape) == 3:
            if k == 0:
                # split head tensor
                bond_d_inp = None
                bond_d_out = B.shape[2]
                m = B.reshape((self.part_d, self.part_d * bond_d_out))

            elif k == self.N-2:
                # split tail tensor
                bond_d_inp = B.shape[0]
                bond_d_out = None
                m = B.reshape((bond_d_inp * self.part_d, self.part_d))

            else:
                raise Exception("invalid head/tail bond tensor shape")

        elif len(B.shape) == 4:
            bond_d_inp = B.shape[0]
            bond_d_out = B.shape[3]
            m = B.reshape((bond_d_inp * self.part_d, self.part_d * bond_d_out))

        else:
            raise Exception("invalid bond tensor shape")

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        # truncate singular values
        # bond_d = cp.argmin(s / s[0] > svd_thresh)
        # s[bond_d:] = 0
        bond_d = u.shape[1]

        if left:
            v = np.einsum("i,ij->ij", s, v)
        else:
            u = np.einsum("ij,j->ij", u, s)

        if k == 0:
            return u.reshape((self.part_d, bond_d)), v.reshape((bond_d, self.part_d, bond_d_out))

        if k == self.N-2:
            return u.reshape((bond_d_inp, self.part_d, bond_d)), v.reshape((bond_d, self.part_d))

        return u.reshape((bond_d_inp, self.part_d, bond_d)), v.reshape((bond_d, self.part_d, bond_d_out))


    def squared_norm(self) -> float:
        """
        Compute the squared norm of the MPS
        """
        T = np.einsum("pi,pj->ij", self._get(0), self._get(0))
        for n in range(1,self.N-1):
            T = np.einsum("ij,ipk,jpl->kl", T, self._get(n), self._get(n))

        T = np.einsum("ij,ip,jp->", T, self._get(self.N-1), self._get(self.N-1))

        return T.item()


    def log_likelihood(self, X : np.ndarray) -> float:
        """
        Min, max, average log-likelihood
        """
        l = self.log_likelihood_samples(X)
        return np.min(l).item(), (np.sum(l).item() / X.shape[0]), np.max(l).item()


    def log_likelihood_samples(self, X : np.ndarray) -> float:
        """
        Compute log-likelihood of each sample (assume mps is normalized)
        """
        return -2 * np.log(abs(self(X)))


    def __call__(self, X : np.ndarray) -> float:
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
