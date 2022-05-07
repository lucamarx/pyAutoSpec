"""
Mps based function compression algorithm
"""
import numpy as np

from math import pi

from .mps import MpsC


def vector2data(v : np.ndarray, x0 : float = 0.0, x1 : float = 1.0) -> np.ndarray:
    pass


def data2vector(X : np.ndarray, x0 : float = 0.0, x1 : float = 1.0) -> np.ndarray:
    """
    """
    theta = 2*pi * (X - x0) / (x1 - x0)
    return np.dstack((np.cos(theta), np.sin(theta)))


class DatasetMps():
    """
    Mps based classifier
    """

    def __init__(self, field_n : int, class_n, max_bond_dim : int = 20, x0 : float = 0.0, x1 : float = 1.0):
        """
        Intialize a classifier

        Parameters:
        -----------

        field_n : int
        the number of fields in the dataset

        class_n : int
        the number of classes

        max_bond_dim : int
        the underlying MPS maximum bond dimension
        """
        if x1 <= x0:
            raise Exception("x0 must be less than x1")

        self.x0, self.x1 = x0, x1
        self.models = [MpsC(field_n, 2, max_bond_dim) for _ in range(class_n)]


    def __repr__(self) -> str:
        return "  DatasetMps({})\n{}".format(len(self.models), self.model.__repr__())


    def __call__(self, X : np.ndarray) -> int:
        """
        Evaluate the class of x

        Parameters:
        -----------

        x : np.ndarray

        Returns:
        --------

        the estimated class
        """
        l = np.dstack([m.log_likelihood(data2vector(X, x0=self.x0, x1=self.x1)) for m in self.models])
        return np.argmax(l, 2)


    def fit(self, X : np.ndarray, y : np.ndarray, learn_rate : float = 0.1, batch_size : int = 32, epochs : int = 10):
        """
        Fit the model to the data

        Parameters:
        -----------

        X : np.ndarray
        y : np.ndarray

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
        for c in range(len(self.models)):
            self.models[c].fit(data2vector(X[y == (c+1)], x0=self.x0, x1=self.x1), learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

        return self
