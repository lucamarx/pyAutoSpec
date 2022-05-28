"""
Density Matrix Renormalization Group learning

This version supports multi-label classification
"""
import numpy as np

from tqdm.auto import tqdm
from typing import List, Tuple

from .dmrg_learning import ContractionCache, _contract_left, _contract_right


def cost(mps, X : np.ndarray, y : np.ndarray) -> float:
    """
    Compute cost function
    """
    f_l, y_l = mps(X), np.eye(mps.class_d)[y]

    return (1/2) * np.sum(np.square(f_l - y_l)).item()


def _move_pivot_r2l(mps, X : np.ndarray, y : np.ndarray, j : int, learn_rate : float, cache : ContractionCache = None) -> int:
    """
    Optimize chain at pivot point j (moving from right to left)

    Gradient of

         1         l                2
    C = --- Σ  Σ (f(X_n) - ẟ(l,y_n))
         2   n  l

    is computed as

               l                    l
    ∇ C = Σ  (f(X_n) - ẟ(l,y_n)) ∇ f(X_n)
     B     n                      B
                                           l
    where                                  │
                     ╭───┐      ╭───┐             ╭───┐      ╭───┐
    ∇ f[a,i,j,b,l] = │ 1 ├─ .. ─┤k-1├─a         b─┤k+2├─ .. ─┤ N │
     B               └─┬─┘      └─┬─┘   i     j   └─┬─┘      └─┬─┘
                       │          │     │     │     │          │
                       ◯          ◯     ◯     ◯     ◯          ◯
                             L        x[k]  x[k+1]       R
    """
    delta = np.eye(mps.class_d)[y]

    if j == len(mps)-1:
        # TAIL
        # merge tensors at j-1 and j (pivot is at j)
        B = np.einsum("ipk,kql->iplq", mps[j-1], mps[j])

        # compute gradient
        L = cache[j-2] if cache is not None else _contract_left(mps, j-2, X)
        f = np.einsum("bi,bp,bq,iplq->bl", L, X[:,j-1,:], X[:,j,:], B)
        G = np.einsum("bi,bp,bq,bl->iplq", L, X[:,j-1,:], X[:,j,:], (f - delta))

        # make SGD step
        B -= learn_rate * G

        # split B tensor moving pivot at j-1
        bond_d_inp = B.shape[0]

        m = B.reshape((bond_d_inp * mps.part_d * mps.class_d, mps.part_d))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        u = np.einsum("ij,j->ij", u, s)

        mps[j-1], mps[j] = u.reshape((bond_d_inp, mps.part_d, mps.class_d, bond_d)), v.reshape((bond_d, mps.part_d))

        if cache is not None:
            cache[j] = np.einsum("ip,bp->bi", mps[j], X[:, j, :])

    elif 1 < j and j < len(mps)-1:
        # MIDDLE
        # merge tensors at j-1 and j (pivot is at j)
        B = np.einsum("ipk,kqlj->iplqj", mps[j-1], mps[j])

        # compute gradient
        L = cache[j-2] if cache is not None else _contract_left(mps, j-2, X)
        R = cache[j+1] if cache is not None else _contract_right(mps, j+1, X)

        f = np.einsum("bi,bp,iplqj,bq,bj->bl", L, X[:,j-1,:], B, X[:,j,:], R)
        G = np.einsum("bi,bp,bq,bj,bl->iplqj", L, X[:,j-1,:], X[:,j,:], R, (f - delta))

        # make SGD step
        B -= learn_rate * G

        # split B tensor moving pivot at j-1
        bond_d_inp, bond_d_out = B.shape[0], B.shape[4]

        m = B.reshape((bond_d_inp * mps.part_d * mps.class_d, mps.part_d * bond_d_out))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        # split B tensor moving pivot at j-1
        bond_d = u.shape[1]

        u = np.einsum("ij,j->ij", u, s)

        mps[j-1], mps[j] = u.reshape((bond_d_inp, mps.part_d, mps.class_d, bond_d)), v.reshape((bond_d, mps.part_d, bond_d_out))

        if cache is not None:
            cache[j] = np.einsum("ipj,bp,bj->bi", mps[j], X[:, j, :], cache[j+1])

    elif j == 1:
        # HEAD
        # merge tensors at j-1 and j (pivot is at j)
        B = np.einsum("pk,kqlj->plqj", mps[j-1], mps[j])

        # compute gradient
        R = cache[j+1] if cache is not None else _contract_right(mps, j+1, X)

        f = np.einsum("plqj,bp,bq,bj->bl", B, X[:,j-1,:], X[:,j,:], R)
        G = np.einsum("bp,bq,bj,bl->plqj", X[:,j-1,:], X[:,j,:], R, (f - delta))

        # make SGD step
        B -= learn_rate * G

        # split B tensor moving pivot at j-1
        bond_d_out = B.shape[3]

        m = B.reshape((mps.part_d * mps.class_d, mps.part_d * bond_d_out))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        u = np.einsum("ij,j->ij", u, s)

        mps[j-1], mps[j] = u.reshape((mps.part_d, mps.class_d, bond_d)), v.reshape((bond_d, mps.part_d, bond_d_out))

        if cache is not None:
            cache[j] = np.einsum("ipj,bp,bj->bi", mps[j], X[:, j, :], cache[j+1])

    else:
        raise Exception("pivot is at head of chain")

    return j-1


def _move_pivot_l2r(mps, X : np.ndarray, y : np.ndarray, j : int, learn_rate : float, cache : ContractionCache = None) -> int:
    """
    Optimize chain at pivot point j (moving from left to right)

    Gradient of

         1         l                2
    C = --- Σ  Σ (f(X_n) - ẟ(l,y_n))
         2   n  l

    is computed as

               l                    l
    ∇ C = Σ  (f(X_n) - ẟ(l,y_n)) ∇ f(X_n)
     B     n                      B
                                           l
    where                                  │
                     ╭───┐      ╭───┐             ╭───┐      ╭───┐
    ∇ f[a,i,j,b,l] = │ 1 ├─ .. ─┤k-1├─a         b─┤k+2├─ .. ─┤ N │
     B               └─┬─┘      └─┬─┘   i     j   └─┬─┘      └─┬─┘
                       │          │     │     │     │          │
                       ◯          ◯     ◯     ◯     ◯          ◯
                             L        x[k]  x[k+1]       R
    """
    delta = np.eye(mps.class_d)[y]

    if j == 0:
        # HEAD
        # merge tensors at j and j+1 (pivot is at j)
        B = np.einsum("plk,kqj->pqlj", mps[j], mps[j+1])

        # compute gradient
        R = cache[j+2] if cache is not None else _contract_right(mps, j+2, X)

        f = np.einsum("pqlj,bp,bq,bj->bl", B, X[:,j,:], X[:,j+1,:], R)
        G = np.einsum("bp,bq,bj,bl->pqlj", X[:,j,:], X[:,j+1,:], R, (f - delta))

        # make SGD step
        B -= learn_rate * G

        # split B tensor moving pivot at j+1
        bond_d_out = B.shape[3]

        m = B.reshape((mps.part_d, mps.part_d * mps.class_d * bond_d_out))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        v = np.einsum("i,ij->ij", s, v)

        mps[j], mps[j+1] = u.reshape((mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, mps.class_d, bond_d_out))

        if cache is not None:
            cache[j] = np.einsum("bp,pi->bi", X[:, j, :], mps[j])

    elif 0 < j and j < len(mps)-2:
        # MIDDLE
        # merge tensors at j and j+1 (pivot is at j)
        B = np.einsum("iplk,kqj->ipqlj", mps[j], mps[j+1])

        # compute gradient
        L = cache[j-1] if cache is not None else _contract_left(mps, j-1, X)
        R = cache[j+2] if cache is not None else _contract_right(mps, j+2, X)

        f = np.einsum("bi,bp,ipqlj,bq,bj->bl", L, X[:,j,:], B, X[:,j+1,:], R)
        G = np.einsum("bi,bp,bq,bj,bl->ipqlj", L, X[:,j,:], X[:,j+1,:], R, (f - delta))

        # make SGD step
        B -= learn_rate * G

        # split B tensor moving pivot at j+1
        bond_d_inp, bond_d_out = B.shape[0], B.shape[4]

        m = B.reshape((bond_d_inp * mps.part_d, mps.part_d * mps.class_d * bond_d_out))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        v = np.einsum("i,ij->ij", s, v)

        mps[j], mps[j+1] = u.reshape((bond_d_inp, mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, mps.class_d, bond_d_out))

        if cache is not None:
            cache[j] = np.einsum("bi,bp,ipj->bj", cache[j-1], X[:, j, :], mps[j])

    elif j == len(mps)-2:
        # TAIL
        # merge tensors at j and j+1 (pivot is at j)
        B = np.einsum("iplk,kq->ipql", mps[j], mps[j+1])

        # compute gradient
        L = cache[j-1] if cache is not None else _contract_left(mps, j-1, X)

        f = np.einsum("bi,bp,bq,ipql->bl", L, X[:,j,:], X[:,j+1,:], B)
        G = np.einsum("bi,bp,bq,bl->ipql", L, X[:,j,:], X[:,j+1,:], (f - delta))

        # make SGD step
        B -= learn_rate * G

        # split B tensor moving pivot at j+1
        bond_d_inp = B.shape[0]

        m = B.reshape((bond_d_inp * mps.part_d, mps.part_d * mps.class_d))

        u, s, v = np.linalg.svd(m, full_matrices=False, compute_uv=True)

        bond_d = u.shape[1]

        v = np.einsum("i,ij->ij", s, v)

        mps[j], mps[j+1] = u.reshape((bond_d_inp, mps.part_d, bond_d)), v.reshape((bond_d, mps.part_d, mps.class_d))

        if cache is not None:
            cache[j] = np.einsum("bi,bp,ipj->bj", cache[j-1], X[:, j, :], mps[j])

    else:
        raise Exception("pivot is at tail of chain")

    return j+1


def fit_classification(mps, X_train : np.ndarray, y_train : np.ndarray, X_valid : np.ndarray = None, y_valid : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10, early_stop : bool = False) -> Tuple[List[float], List[float]]:
    """Fit the MPS to the data

    0. for each epoch
    1.  sample a random mini-batch from X
    2.  sweep right → left (left → right)
    3.   contract A^k and A^(k+1) into B^k
    4.   evaluate gradients for mini-batch
    5.   update B^k
    6.   split B^k with SVD
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

    early_stop : bool stop as soon as overfitting is detected (needs a
    validation dataset)

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
            n = len(mps)-1
            while True:
                n = _move_pivot_r2l(mps, X_batch, y_batch, n, learn_rate, cache)
                if n == 0:
                    break

            # left to right pass
            n = 0
            while True:
                n = _move_pivot_l2r(mps, X_batch, y_batch, n, learn_rate, cache)
                if n == len(mps)-1:
                    break

        train_cost = cost(mps, X_train, y_train)
        train_costs.append(train_cost)

        if X_valid is not None and y_valid is not None:
            valid_cost = cost(mps, X_valid, y_valid)
            valid_costs.append(valid_cost)

            mavg = 0 if len(valid_mavg) == 0 else sum(valid_mavg) / len(valid_mavg)
            if early_stop and len(valid_mavg) > 15 and valid_cost > mavg:
                print("            overfitting detected: validation score is rising over moving average")
                print("            training interrupted")
                break

            if epoch % 10 == 0:
                print("epoch {:4d}: train {:.2f} | valid {:.2f}".format(epoch, train_cost, valid_cost))

            valid_mavg.append(valid_cost)
            if len(valid_mavg) > 20:
                valid_mavg.pop(0)

        else:
            if epoch % 10 == 0:
                print("epoch {:4d}: {:.2f}".format(epoch, train_cost))

    return train_costs, valid_costs
