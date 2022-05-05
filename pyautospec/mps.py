"""
Matrix product state class
"""
import numpy as np

from typing import List, Tuple
from tqdm.auto import tqdm


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


    def __repr__(self) -> str:
        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

  particle dim:  {}
  max bond dim:  {}
        """.format(self.N, self.part_d, self.max_bond_d)


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

        or at the head/tail of the chain

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

        or at the head/tail of the chain

          ╭────┐     ╭───┐ ╭───┐
          │    ├─ =  │ 0 ├─┤ 1 ├─
          └┬──┬┘     └─┬─┘ └─┬─┘

          ╭────┐     ╭───┐ ╭───┐
         ─┤    │  = ─┤N-1├─┤ N │
          └┬──┬┘     └─┬─┘ └─┬─┘
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


    def _contract_left(self, k : int, X :np.ndarray) -> np.ndarray:
        """
        Contract MPS from left stopping at k < N-1

        ╭───┐ ╭───┐       ╭───┐
        │ 1 ├─┤ 2 ├─ ... ─┤ k ├─
        └─┬─┘ └─┬─┘       └─┬─┘
          ◯     ◯           ◯
        """
        L = np.einsum("bp,pj->bj", X[:,0,:], self._get(0))
        for n in range(1,k+1):
            L = np.einsum("bi,bp,ipj->bj", L, X[:,n,:], self._get(n))
        return L


    def _contract_right(self, k : int, X : np.ndarray) -> np.ndarray:
        """
        Contract MPS from right stopping at k > 0

         ╭───┐       ╭───┐ ╭───┐
        ─┤ k ├─ ... ─┤N-1├─┤ N │
         └─┬─┘       └─┬─┘ └─┬─┘
           ◯           ◯     ◯
        """
        R = np.einsum("ip,bp->bi", self._get(self.N-1), X[:,self.N-1,:])
        for n in reversed(range(k,self.N-1)):
            R = np.einsum("ipj,bp,bj->bi", self._get(n), X[:,n,:], R)
        return R


    def _initialize_cache(self, X : np.ndarray):
        """
        Initialize left contractions cache
        """
        self.cache[0] = np.einsum("bp,pj->bj", X[:, 0, :], self._get(0))
        for n in range(1, self.N-1):
            self.cache[n] = np.einsum("bi,bp,ipj->bj", self.cache[n-1], X[:, n, :], self._get(n))


    def _gradient(self, k : int, X : np.ndarray, y : np.ndarray, use_cache : bool = False) -> float:
        """
        Let

              1                   2
        C =  --- Σ  (f(X_n) - y_n)
              N   n

        evaluate

               2
        ∇ C = --- Σ  (f(X_n) - y_n) ∇ f(X_n)
         B     N   n                 B

        where
                       ╭───┐      ╭───┐            ╭───┐      ╭───┐
        ∇ f[a,i,j,b] = │ 1 ├─ .. ─┤k-1├─a        b─┤k+2├─ .. ─┤ N │
         B             └─┬─┘      └─┬─┘   i    j   └─┬─┘      └─┬─┘
                         │          │     │    │     │          │
                         ◯          ◯     ◯    ◯     ◯          ◯
                               L        x[k] x[k+1]       R
        """
        batch_d = X.shape[0]

        if k == 0:
            R = self.cache[k+2] if use_cache else self._contract_right(k+2, X)

            # avoid mps re-evaluation
            w = np.einsum("ipj,bp,bj->bi", self._get(1), X[:,k+1,:], R)
            v = np.einsum("pi,bp,bi->b", self._get(0), X[:,k,:], w)

            # perform tensor products
            d = np.einsum("bp,bq,bi,b->pqi", X[:,k,:], X[:,k+1,:], R, (v - y))

        elif k == self.N-2:
            L = self.cache[k-1] if use_cache else self._contract_left(k-1, X)

            # avoid mps re-evaluation
            u = np.einsum("bi,bp,ipj->bj", L, X[:,k,:], self._get(k))
            v = np.einsum("bj,bp,jp->b", u, X[:,k+1,:], self._get(k+1))

            # perform tensor products
            d = np.einsum("bi,bp,bq,b->ipq", L, X[:,k,:], X[:,k+1,:], (v - y))

        elif 0 < k and k < (self.N-2):
            L = self.cache[k-1] if use_cache else self._contract_left(k-1, X)
            u = np.einsum("bi,bp,ipj->bj", L, X[:,k,:], self._get(k))

            R = self.cache[k+2] if use_cache else self._contract_right(k+2, X)
            w = np.einsum("ipj,bp,bj->bi", self._get(k+1), X[:,k+1,:], R)

            # avoid mps re evaluation
            v = np.einsum("bi,bi->b", u, w)

            # perform tensor products
            d = np.einsum("bi,bp,bq,bj,b->ipqj", L, X[:,k,:], X[:,k+1,:], R, (v - y))

        else:
            raise Exception("invalid k")

        return 2 * d / batch_d


    def _move_pivot(self, X : np.ndarray, y : np.ndarray, n : int, learn_rate : float, direction : str) -> int:
        """
        Optimize chain at pivot point n (moving in direction)
        """
        B = self._merge(n)
        G = self._gradient(n, X, y, use_cache=True)

        B -= learn_rate * G

        A1, A2 = self._split(n, B, left=(direction == "left2right"))

        self._set(n, A1)
        self._set(n+1, A2)

        if direction == "right2left":
            if n+1 == self.N-1:
                self.cache[n+1] = np.einsum("ip,bp->bi", self._get(n+1), X[:, n+1, :])
            else:
                self.cache[n+1] = np.einsum("ipj,bp,bj->bi", self._get(n+1), X[:, n+1, :], self.cache[n+2])

            return n-1

        if direction == "left2right":
            if n == 0:
                self.cache[n] = np.einsum("bp,pi->bi", X[:, n, :], self._get(n))
            else:
                self.cache[n] = np.einsum("bi,bp,ipj->bj", self.cache[n-1], X[:, n, :], self._get(n))

            return n+1

        raise Exception("invalid direction")


    def cost(self, X : np.ndarray, y : np.ndarray) -> Tuple[float, float, float]:
        """
        Compute cost function
        """
        c = np.square(self(X) - y)
        return np.min(c).item(), (np.sum(c).item() / X.shape[0]), np.max(c).item()


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

        T = np.einsum("bp,pj->bj", X[:,0,:], self._get(0))
        for n in range(1,self.N-1):
            T = np.einsum("bi,bp,ipj->bj", T, X[:,n,:], self._get(n))

        T = np.einsum("bi,bp,ip->b", T, X[:,self.N-1,:], self._get(self.N-1))

        return T


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
        if len(X.shape) != 3:
            raise Exception("invalid data")

        if X.shape[1] != self.N:
            raise Exception("invalid shape for X (wrong particle number)")

        if X.shape[2] != self.part_d:
            raise Exception("invalid shape for X (wrong particle dimension)")

        for epoch in tqdm(range(1, epochs+1)):
            for _ in range(1, int(X.shape[0] / batch_size)):
                batch = np.random.randint(0, high=X.shape[0], size=batch_size)
                X_batch, y_batch = X[batch], y[batch]

                self._initialize_cache(X_batch)

                # right to left pass
                n = self.N-2
                while True:
                    n = self._move_pivot(X_batch, y_batch, n, learn_rate, "right2left")
                    if n == -1:
                        break

                # left to right pass
                n = 0
                while True:
                    n = self._move_pivot(X_batch, y_batch, n, learn_rate, "left2right")
                    if n == self.N-1:
                        break

            if epoch % 10 == 0:
                print("epoch {:4d}: min={:.2f} avg={:.2f} max={:.2f}".format(epoch, *self.cost(X, y)))

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
        self.mps = Mps(N, alphabet, max_bond_d)


    def __repr__(self) -> str:
        return """
  ╭───┐ ╭───┐       ╭───┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

  alphabet size:  {}
   max bond dim:  {}
        """.format(self.mps.N, self.mps.part_d, self.mps.max_bond_d)


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


    def fit(self, X : List[List[int]], y : List[float], learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
        """
        Fit the MPS to the data
        """
        self.mps.fit(self._one_hot(X), np.array(y), learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)
        return self
