"""
Mps based function compression algorithm
"""
import numpy as np

from .mps import MpsR


def data2vector(X : np.ndarray, x0 : np.ndarray, x1 : np.ndarray) -> np.ndarray:
    """
    Convert data points into 2-dim vectors
    """
    theta = 2 * np.pi * (X - x0) / (x1 - x0)

    if np.any(theta < 0) or np.any(theta > 2*np.pi):
        raise Exception("out of range")

    return np.dstack((np.cos(theta), np.sin(theta)))


class DatasetMps():
    """
    Mps based classification/regression
    """

    def __init__(self, field_n : int, max_bond_dim : int = 20, class_n : int = None):
        """
        Intialize a classification/regression model

        Parameters:
        -----------

        field_n : int
        the number of fields in the dataset

        max_bond_dim : int
        the underlying MPS maximum bond dimension

        class_n : int
        the number of classes (None for regression)
        """
        self.x0, self.x1 = None, None

        self.field_n = field_n

        if class_n is not None:
            self.classification_model = [MpsR(field_n, 2, max_bond_dim) for _ in range(class_n)]
            self.regression_model = None
        else:
            self.regression_model = MpsR(field_n, 2, max_bond_dim)
            self.classification_model = None


    def __repr__(self) -> str:
        if self.classification_model is not None:
            return "  DatasetMps(classification)\n{}".format("\n".join([m.__repr__() for m in self.classification_model]))
        else:
            return "  DatasetMps(regression)\n{}".format(self.regression_model)


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
        if self.x0 is None or self.x1 is None:
            raise Exception("the model has not been trained yet")

        if self.classification_model is not None:
            l = np.dstack([np.abs(1.0 - m(data2vector(X, x0=self.x0, x1=self.x1))) for m in self.classification_model])
            return np.argmin(l, 2)
        else:
            return self.regression_model(data2vector(X, x0=self.x0, x1=self.x1))


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


    def fit(self, X : np.ndarray, y : np.ndarray, learn_rate : float = 0.1, batch_size : int = 10, epochs : int = 50):
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
        if X.shape[0] != y.shape[0]:
            raise Exception("X and y have different sizes")

        if X.shape[1] != self.field_n:
            raise Exception("invalid number of fields")

        self.x0, self.x1 = 0.9 * np.min(X, axis=0), 1.1 * np.max(X, axis=0)

        if self.classification_model is not None:
            # train the classification models for each class label
            for cls in range(len(self.classification_model)):
                self.classification_model[cls].fit(data2vector(X, x0=self.x0, x1=self.x1), (y == cls).astype(int), learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

        else:
            # train the regression model
            self.regression_model.fit(data2vector(X, x0=self.x0, x1=self.x1), y, learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

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

        the accuracy
        """
        if self.classification_model is not None:
            t = self(X) - y
            return np.average((t == 0).astype(int))
        else:
            return np.average(np.square(self(X) - y))
