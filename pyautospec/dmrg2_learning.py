"""
Density Matrix Renormalization Group learning

This version supports multi-label classification
"""
import numpy as np

from typing import Tuple
from tqdm.auto import tqdm

from .dmrg_learning import ContractionCache, _contract_left, _contract_right


def cost(mps, X : np.ndarray, y : np.ndarray) -> Tuple[float, float, float]:
    """
    Compute cost function
    """
    f_l, y_l = mps(X), np.eye(mps.labels)[y]

    return (1/2) * np.sum(np.square(f_l - y_l)).item()


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


def _move_pivot_r2l(mps, X : np.ndarray, y : np.ndarray, j : int, learn_rate : float, cache : ContractionCache = None) -> int:
    """
    Optimize chain at pivot point j (moving right to left)
    """
    # merge tensors at j-1 and j (pivot is at j)
    if j == mps.N-1:
        B = np.einsum("ipj,jql->ipql", mps[j-1], mps[j])
    elif 0 < j and j < mps.N-1:
        B = np.einsum("ipj,jqkl->ipqkl", mps[j-1], mps[j])
    else:
        raise Exception("pivot is at head of chain")

    # compute gradient
    G = _gradient(mps, j, X, y, cache=cache)

    # make SGD step
    B -= learn_rate * G

    # split B tensor moving pivot at j-1
    if j == mps.N-1:
        bond_d_inp = B.shape[0]

        m = B.reshape((bond_d_inp * mps.part_d * mps.labels, mps.part_d))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        s_sqrt = np.sqrt(s)
        u = np.einsum("ij,j->ij", u, s_sqrt)
        v = np.einsum("i,ij->ij", s_sqrt, v)

        mps[j-1], mps[j] = u.reshape((bond_d_inp, mps.part_d, bond_d, mps.labels)), v.reshape((bond_d, mps.part_d))

        if cache is not None:
            cache[j] = np.einsum("ip,bp->bi", mps[j], X[:, j, :])

    elif 0 < j and j < mps.N-1:
        bond_d_inp, bond_d_out = B.shape[0], B.shape[3]

        m = B.reshape((bond_d_inp * mps.part_d * mps.labels, mps.part_d * bond_d_out))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        s_sqrt = np.sqrt(s)
        u = np.einsum("ij,j->ij", u, s_sqrt)
        v = np.einsum("i,ij->ij", s_sqrt, v)

        mps[j-1], mps[j] = u.reshape((bond_d_inp, mps.part_d, bond_d, mps.labels)), v.reshape((bond_d, mps.part_d, bond_d_out))

        if cache is not None:
            cache[j] = np.einsum("ipj,bp,bj->bi", mps[j], X[:, j, :], cache[j+1])

    return j-1


def _move_pivot_l2r(mps, X : np.ndarray, y : np.ndarray, j : int, learn_rate : float, cache : ContractionCache = None) -> int:
    """
    Optimize chain at pivot point n (moving left to right)
    """
    # merge tensors at j and j+1 (pivot is at j)
    if j == 0:
        B = np.einsum("pjl,jqk->pqkl", mps[j], mps[j+1])
    elif 0 < j and j < mps.N-1:
        B = np.einsum("ipjl,jqk->ipqkl", mps[j], mps[j+1])
    else:
        raise Exception("pivot is at tail of chain")

    # compute gradient
    G = _gradient(mps, j, X, y, cache=cache)

    # make SGD step
    B -= learn_rate * G

    # split B tensor moving pivot at j+1
    if j == 0:
        bond_d_out = B.shape[2]

        m = B.reshape((mps.part_d, mps.part_d * bond_d_out * mps.labels))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        s_sqrt = np.sqrt(s)
        u = np.einsum("ij,j->ij", u, s_sqrt)
        v = np.einsum("i,ij->ij", s_sqrt, v)

        mps[j], mps[j+1] = u.reshape((mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, bond_d_out, mps.labels))

        if cache is not None:
            cache[j] = np.einsum("bp,pi->bi", X[:, j, :], mps[j])

    elif 0 < j and j < mps.N-1:
        bond_d_inp, bond_d_out = B.shape[0], B.shape[3]

        m = B.reshape((bond_d_inp * mps.part_d, mps.part_d * bond_d_out * mps.labels))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        s_sqrt = np.sqrt(s)
        u = np.einsum("ij,j->ij", u, s_sqrt)
        v = np.einsum("i,ij->ij", s_sqrt, v)

        mps[j], mps[j+1] = u.reshape((bond_d_inp, mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, bond_d_out, mps.labels))

        if cache is not None:
            cache[j] = np.einsum("bi,bp,ipj->bj", cache[j-1], X[:, j, :], mps[j])

    return j+1


def fit_classification(mps, X_train : np.ndarray, y_train : np.ndarray, X_valid : np.ndarray = None, y_valid : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10, early_stop : bool = False):
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

    if X_train.shape[1] != len(mps):
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
            n = mps.N-1
            while True:
                n = _move_pivot_r2l(mps, X_batch, y_batch, n, learn_rate, cache)
                if n == 0:
                    break

            # left to right pass
            n = 0
            while True:
                n = _move_pivot_l2r(mps, X_batch, y_batch, n, learn_rate, cache)
                if n == mps.N-1:
                    break

        train_cost = cost(mps, X_train, y_train)
        train_costs.append(train_cost)

        if X_valid is not None and y_valid is not None:
            valid_cost = cost(mps, X_valid, y_valid)
            valid_costs.append(valid_cost)

            mavg = 0 if len(valid_mavg) == 0 else sum(valid_mavg) / len(valid_mavg)
            if early_stop and len(valid_mavg) > 15 and valid_cost[0] > mavg:
                print("            overfitting detected: validation score is rising over moving average")
                print("            training interrupted")
                break

            if epoch % 10 == 0:
                print("epoch {:4d}: train {:.2f} | valid {:.2f}".format(epoch, train_cost, valid_cost))

            valid_mavg.append(valid_cost[0])
            if len(valid_mavg) > 20:
                valid_mavg.pop(0)

        else:
            if epoch % 10 == 0:
                print("epoch {:4d}: {:.2f}".format(epoch, train_cost))

    return train_costs, valid_costs
