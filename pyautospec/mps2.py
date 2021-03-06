"""
Matrix product state class
"""
import numpy as np

from typing import List

from .plots import training_chart
from .dmrg2_learning import cost, fit


class Mps2:
    """
    Matrix Product State for regression/classification (with pivot)

                              │
    ╭───┐ ╭───┐ ╭───┐       ╭─┴─┐
    │ 1 ├─┤ 2 ├─┤ 3 ├─ ... ─┤ N │
    └─┬─┘ └─┬─┘ └─┬─┘       └─┬─┘
      │     │     │           │
    """

    def __init__(self, N : int, part_d : int = 2, max_bond_d : int = 20, class_d : int = 2, model_type : str = "classification"):
        """
        Initialize a random matrix product state, positioning the pivot at the tail

        Parameters:
        -----------

        N : int
        number of particles

        part_d : int
        particle dimension

        max_bond_d : int
        maximum bond dimension

        class_d : int
        number of classes

        model_type : str
        the model type: regression or classification
        """
        if N < 3:
            raise Exception("chain too short")

        if part_d < 2:
            raise Exception("particle dimension must be at least 2")

        if model_type == "classification" and class_d < 2:
            raise Exception("class dimension must be at least 2 for classification")

        if model_type not in ["regression", "classification"]:
            raise Exception("model_type must be either 'regression' or 'classification'")

        self.N = N
        self.part_d = part_d
        self.class_d = class_d
        self.max_bond_d = max_bond_d
        self.model_type = model_type

        # start with small bond dimension
        bond_d = 2

        # setup tensor container (position pivot at the tail)
        self.mps = [np.random.rand(*s) for s in [(part_d,bond_d)] + [(bond_d,part_d,bond_d)]*(N-2) + [(bond_d,part_d,class_d)]]
        self.pivot = N-1

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
                self[n+1] = np.einsum("i,ij,jql->iql", s, v, self[n+1])

        # normalize
        t = self[N-1]
        norm = np.sqrt(np.einsum("ipl,ipl->", t, t))
        self[N-1] = t / norm

        # initialize training/validation costs
        self.train_costs, self.valid_costs = [], []


    def __repr__(self) -> str:
        bond_d = max([self[n].shape[-1] for n in range(len(self)-1)])

        return """
  ╭───┐ ╭───┐       ╭─┴─┐
  │ 1 ├─┤ 2 ├─ ... ─┤{:3d}│
  └─┬─┘ └─┬─┘       └─┬─┘

  particle dim: {:3d}
     class dim: {:3d}
      bond dim: {:3d} (max: {:d})
          type: {}
        """.format(self.N, self.part_d, self.class_d, bond_d, self.max_bond_d, self.model_type)


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
            if len(m.shape) == 2:
                self.mps[n] = m[:, 0:self.max_bond_d]
            elif len(m.shape) == 3:
                self.mps[n] = m[:, :, 0:self.max_bond_d]
                self.pivot = n
            else:
                raise Exception("invalid tensor")

        elif n == self.N-1:
            if len(m.shape) == 2:
                self.mps[n] = m[0:self.max_bond_d, :]
            elif len(m.shape) == 3:
                self.mps[n] = m[0:self.max_bond_d, :, :]
                self.pivot = n
            else:
                raise Exception("invalid tensor")

        else:
            if len(m.shape) == 3:
                self.mps[n] = m[0:self.max_bond_d, :, 0:self.max_bond_d]
            elif len(m.shape) == 4:
                self.mps[n] = m[0:self.max_bond_d, :, :, 0:self.max_bond_d]
                self.pivot = n
            else:
                raise Exception("invalid tensor")


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
        Evaluate MPS on batch X[b,n,p] (assuming the pivot is at the tail)

        ╭───┐ ╭───┐       ╭─┴─┐
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

        T = np.einsum("bi,bp,ipl->bl", T, X[:,self.N-1,:], self[self.N-1])

        return T


    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Predict the class of X

        Parameters:
        -----------

        X : np.ndarray
        a batch of N part_d dimensional vectors

        Returns:
        --------

        the predicted class/value for the batch X
        """
        if self.model_type == "classification":
            return np.argmax(self(X), axis=1)
        else:
            return self(X)


    def cost(self, X : np.ndarray, y : np.ndarray) -> float:
        """
        Compute cost function
        """
        return cost(self, self.model_type, X, y)


    def fit(self, X_train : np.ndarray, y_train : np.ndarray, X_valid : np.ndarray = None, y_valid : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
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
        """
        self.train_costs, self.valid_costs = fit(self, self.model_type, X_train, y_train, X_valid, y_valid, learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

        return self


    def training_chart(self):
        """
        Plots training/validation costs
        """
        if len(self.train_costs) == 0:
            raise Exception("the model has not been trained yet")

        training_chart(self.train_costs, self.valid_costs)
