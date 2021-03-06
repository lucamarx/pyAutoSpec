"""
Density Matrix Renormalization Group learning
"""
import numpy as np

from typing import Tuple
from tqdm.auto import tqdm


class ContractionCache():
    """
    Stores contractions during DMRG sweeps to avoid recalculations
    """

    def __init__(self, mps, X : np.ndarray):
        self.mps = mps
        self.cache = [None] * len(mps)

        self.cache[0] = np.einsum("bp,pj->bj", X[:, 0, :], mps[0])
        for n in range(1, mps.N-1):
            self.cache[n] = np.einsum("bi,bp,ipj->bj", self.cache[n-1], X[:, n, :], mps[n])


    def __len__(self):
        return len(self.cache)


    def __getitem__(self, n):
        return self.cache[n]


    def __setitem__(self, n, v):
        self.cache[n] = v


def cost(mps, X : np.ndarray, y : np.ndarray) -> Tuple[float, float, float]:
    """
    Compute cost function
    """
    c = np.square(mps(X) - y)
    return np.average(c).item(), np.std(c).item()


def squared_norm(mps) -> float:
    """
    Compute the squared norm of the MPS
    """
    T = np.einsum("pi,pj->ij", mps[0], mps[0])
    for n in range(1,mps.N-1):
        T = np.einsum("ij,ipk,jpl->kl", T, mps[n], mps[n])

    T = np.einsum("ij,ip,jp->", T, mps[mps.N-1], mps[mps.N-1])

    return T.item()


def _merge(mps, k : int) -> np.ndarray:
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
    if 0 < k and k+1 < mps.N-1:
        return np.einsum("ipj,jqk->ipqk", mps[k], mps[k+1])

    if k == 0:
        return np.einsum("pj,jqk->pqk", mps[0], mps[1])

    if k+1 == mps.N-1:
        return np.einsum("ipj,jq->ipq", mps[mps.N-2], mps[mps.N-1])

    raise Exception("invalid k")


def _split(mps, k : int, B : np.ndarray, left : bool = True) -> np.ndarray:
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
            m = B.reshape((mps.part_d, mps.part_d * bond_d_out))

        elif k == mps.N-2:
            # split tail tensor
            bond_d_inp = B.shape[0]
            bond_d_out = None
            m = B.reshape((bond_d_inp * mps.part_d, mps.part_d))

        else:
            raise Exception("invalid head/tail bond tensor shape")

    elif len(B.shape) == 4:
        bond_d_inp = B.shape[0]
        bond_d_out = B.shape[3]
        m = B.reshape((bond_d_inp * mps.part_d, mps.part_d * bond_d_out))

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
        return u.reshape((mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, bond_d_out))

    if k == mps.N-2:
        return u.reshape((bond_d_inp, mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d))

    return u.reshape((bond_d_inp, mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, bond_d_out))


def _contract_left(mps, k : int, X :np.ndarray) -> np.ndarray:
    """
    Contract MPS from left stopping at k < N-1

    ╭───┐ ╭───┐       ╭───┐
    │ 1 ├─┤ 2 ├─ ... ─┤ k ├─
    └─┬─┘ └─┬─┘       └─┬─┘
      ◯     ◯           ◯
    """
    L = np.einsum("bp,pj->bj", X[:,0,:], mps[0])
    for n in range(1,k+1):
        L = np.einsum("bi,bp,ipj->bj", L, X[:,n,:], mps[n])
    return L


def _contract_right(mps, k : int, X : np.ndarray) -> np.ndarray:
    """
    Contract MPS from right stopping at k > 0

     ╭───┐       ╭───┐ ╭───┐
    ─┤ k ├─ ... ─┤N-1├─┤ N │
     └─┬─┘       └─┬─┘ └─┬─┘
       ◯           ◯     ◯
    """
    R = np.einsum("ip,bp->bi", mps[mps.N-1], X[:,mps.N-1,:])
    for n in reversed(range(k,mps.N-1)):
        R = np.einsum("ipj,bp,bj->bi", mps[n], X[:,n,:], R)
    return R


def _gradient(mps, k : int, X : np.ndarray, y : np.ndarray, cache : ContractionCache = None) -> float:
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
        R = cache[k+2] if cache is not None else _contract_right(mps, k+2, X)

        # avoid mps re-evaluation
        w = np.einsum("ipj,bp,bj->bi", mps[1], X[:,k+1,:], R)
        v = np.einsum("pi,bp,bi->b", mps[0], X[:,k,:], w)

        # perform tensor products
        d = np.einsum("bp,bq,bi,b->pqi", X[:,k,:], X[:,k+1,:], R, (v - y))

    elif k == mps.N-2:
        L = cache[k-1] if cache is not None else _contract_left(mps, k-1, X)

        # avoid mps re-evaluation
        u = np.einsum("bi,bp,ipj->bj", L, X[:,k,:], mps[k])
        v = np.einsum("bj,bp,jp->b", u, X[:,k+1,:], mps[k+1])

        # perform tensor products
        d = np.einsum("bi,bp,bq,b->ipq", L, X[:,k,:], X[:,k+1,:], (v - y))

    elif 0 < k and k < (mps.N-2):
        L = cache[k-1] if cache is not None else _contract_left(mps, k-1, X)
        u = np.einsum("bi,bp,ipj->bj", L, X[:,k,:], mps[k])

        R = cache[k+2] if cache is not None else _contract_right(mps, k+2, X)
        w = np.einsum("ipj,bp,bj->bi", mps[k+1], X[:,k+1,:], R)

        # avoid mps re evaluation
        v = np.einsum("bi,bi->b", u, w)

        # perform tensor products
        d = np.einsum("bi,bp,bq,bj,b->ipqj", L, X[:,k,:], X[:,k+1,:], R, (v - y))

    else:
        raise Exception("invalid k")

    return 2 * d / batch_d


def _move_pivot(mps, X : np.ndarray, y : np.ndarray, n : int, learn_rate : float, direction : str, cache : ContractionCache = None) -> int:
    """
    Optimize chain at pivot point n (moving in direction)
    """
    B = _merge(mps, n)

    G = _gradient(mps, n, X, y, cache=cache)

    B -= learn_rate * G

    A1, A2 = _split(mps, n, B, left=(direction == "left2right"))

    mps[n]   = A1
    mps[n+1] = A2

    if direction == "right2left":
        if cache is not None:
            if n+1 == mps.N-1:
                cache[n+1] = np.einsum("ip,bp->bi", mps[n+1], X[:, n+1, :])
            else:
                cache[n+1] = np.einsum("ipj,bp,bj->bi", mps[n+1], X[:, n+1, :], cache[n+2])

        return n-1

    if direction == "left2right":
        if cache is not None:
            if n == 0:
                cache[n] = np.einsum("bp,pi->bi", X[:, n, :], mps[n])
            else:
                cache[n] = np.einsum("bi,bp,ipj->bj", cache[n-1], X[:, n, :], mps[n])

        return n+1

    raise Exception("invalid direction: {}".format(direction))


def fit_regression(mps, X_train : np.ndarray, y_train : np.ndarray, X_valid : np.ndarray = None, y_valid : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10, early_stop : bool = False):
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

    Returns:
    --------

    The training and validation costs
    """
    if len(X_train.shape) != 3:
        raise Exception("invalid data")

    if X_train.shape[0] != y_train.shape[0]:
        raise Exception("invalid shape for X,y (wrong sample number)")

    if X_train.shape[1] != mps.N:
        raise Exception("invalid shape for X (wrong particle number)")

    if X_train.shape[2] != mps.part_d:
        raise Exception("invalid shape for X (wrong particle dimension)")

    train_costs, valid_costs, valid_mavg = [], [], []
    for epoch in tqdm(range(1, epochs+1)):
        for _ in range(1, int(X_train.shape[0] / batch_size)):
            batch = np.random.randint(0, high=X_train.shape[0], size=batch_size)
            X_batch, y_batch = X_train[batch], y_train[batch]

            cache = ContractionCache(mps, X_batch)

            # right to left pass
            n = mps.N-2
            while True:
                n = _move_pivot(mps, X_batch, y_batch, n, learn_rate, "right2left", cache)
                if n == -1:
                    break

            # left to right pass
            n = 0
            while True:
                n = _move_pivot(mps, X_batch, y_batch, n, learn_rate, "left2right", cache)
                if n == mps.N-1:
                    break

        train_cost = cost(mps, X_train, y_train)
        train_costs.append(train_cost[0])

        if X_valid is not None and y_valid is not None:
            valid_cost = cost(mps, X_valid, y_valid)
            valid_costs.append(valid_cost[0])

            mavg = 0 if len(valid_mavg) == 0 else sum(valid_mavg) / len(valid_mavg)
            if early_stop and len(valid_mavg) > 15 and valid_cost[0] > mavg:
                print("            overfitting detected: validation score is rising over moving average")
                print("            training interrupted")
                break

            if epoch % 10 == 0:
                print("epoch {:4d}: train avg={:.2f} std={:.2f} | valid avg={:.2f} std={:.2f}".format(epoch, *train_cost, *valid_cost))

            valid_mavg.append(valid_cost[0])
            if len(valid_mavg) > 20:
                valid_mavg.pop(0)

        else:
            if epoch % 10 == 0:
                print("epoch {:4d}: avg={:.2f} std={:.2f}".format(epoch, *train_cost))

    return train_costs, valid_costs
