"""
Mps based classification/regression
"""
import numpy as np

from typing import Tuple

from .mps2 import Mps2
from .plots import mps_entanglement_entropy_chart


class DatasetMps():
    """
    Mps based classification/regression
    """

    def __init__(self, field_n : int, x0 : np.ndarray = None, x1 : np.ndarray = None, max_bond_d : int = 20, class_n : int = None):
        """
        Intialize a classification/regression model

        Parameters:
        -----------

        field_n : int
        the number of fields in the dataset

        x0 : np.ndarray
        x1 : np.ndarray
        the data ranges

        max_bond_dim : int
        the underlying MPS maximum bond dimension

        class_n : int
        the number of classes (None for regression)
        """
        self.field_n = field_n

        if x0 is None or x1 is None:
            x0, x1 = np.zeros((field_n, )), np.ones((field_n, ))

        self.x0, self.x1 = x0, x1

        if class_n is not None:
            self.model = Mps2(field_n, part_d=2, max_bond_d=max_bond_d, class_d=class_n, model_type="classification")
        else:
            self.model = Mps2(field_n, part_d=2, max_bond_d=max_bond_d, class_d=1, model_type="regression")


    def __repr__(self) -> str:
        return "  DatasetMps\n{}".format(self.model.__repr__())


    def _encode(self, X : np.ndarray) -> np.ndarray:
        """
        Encode data points into 2-dim vectors
        """
        theta = (np.pi/2) * (X - self.x0) / (self.x1 - self.x0)

        if np.any(theta < 0) or np.any(theta > np.pi/2):
            raise Exception("out of range")

        return np.dstack((np.cos(theta), np.sin(theta)))


    def __call__(self, X : np.ndarray) -> int:
        """
        Evaluate the model at X

        Parameters:
        -----------

        X : np.ndarray

        Returns:
        --------

        the estimated class/value
        """
        return self.model.predict(self._encode(X))


    def predict(self, X : np.ndarray) -> np.ndarray:
        """
        Evaluate the model at X

        Parameters:
        -----------

        X : np.ndarray

        Returns:
        --------

        the estimated class/value
        """
        return self(X)


    def entanglement_entropy_chart(self):
        """
        Plot entanglement entropy chart
        """
        mps_entanglement_entropy_chart(self.model)


    def fit(self, X_train : np.ndarray, y_train : np.ndarray, X_valid : np.ndarray = None, y_valid : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 10, epochs : int = 50, callback = None):
        """
        Fit the model to the data

        Parameters:
        -----------

        X_train : np.ndarray
        y_train : np.ndarray
        the training dataset

        X_valid : np.ndarray
        y_valid : np.ndarray
        the optional validation dataset

        learn_rate : float
        the learning rate

        batch_size : int
        the batch size used at each step

        epochs : int
        the number of epochs

        early_stop : bool
        stop as soon as overfitting is detected (needs a validation dataset)

        callback: function(mps, epoch)
        it is called at each dmrg training epoch

        Returns:
        --------

        The object itself
        """
        if X_train.shape[0] != y_train.shape[0]:
            raise Exception("X and y have different sizes")

        if X_train.shape[1] != self.field_n:
            raise Exception("invalid number of fields")

        X_train = self._encode(X_train)

        if X_valid is not None and y_valid is not None:
            X_valid = self._encode(X_valid)

        self.model.fit(X_train, y_train, X_valid, y_valid, learn_rate=learn_rate, batch_size=batch_size, epochs=epochs, callback=callback)

        return self


    def score(self, X : np.ndarray, y : np.ndarray) -> float:
        """
        Model score on test data

        Parameters:
        -----------

        X : np.ndarray
        y : np.ndarray

        Returns:
        --------

        the accuracy/cost
        """
        if self.model.model_type == "classification":
            t = self(X) - y
            return np.average((t == 0).astype(int))
        else:
            return np.average(np.square(self(X) - y))


    def paths_weights(self, X : np.ndarray, threshold : float = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Enumerate all paths contributing to the final value
        """
        if self.model.model_type == "classification":
            return self.model.paths_weights(self._encode(X)[0,:], l=self.predict(X)[0], threshold=threshold)
        else:
            return self.model.paths_weights(self._encode(X)[0,:], l=0, threshold=threshold)
