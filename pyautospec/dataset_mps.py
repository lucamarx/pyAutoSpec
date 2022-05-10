"""
Mps based classification/regression
"""
import numpy as np

from .mps import Mps


def data2vector(X : np.ndarray, x0 : np.ndarray, x1 : np.ndarray) -> np.ndarray:
    """
    Convert data points into 2-dim vectors
    """
    theta = (np.pi/2) * (X - x0) / (x1 - x0)

    if np.any(theta < 0) or np.any(theta > np.pi/2):
        raise Exception("out of range")

    return np.dstack((np.cos(theta), np.sin(theta)))


class DatasetMps():
    """
    Mps based classification/regression
    """

    def __init__(self, field_n : int, x0 : np.ndarray = None, x1 : np.ndarray = None, max_bond_dim : int = 20, class_n : int = None):
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
            self.classification_model = [Mps(field_n, 2, max_bond_dim) for _ in range(class_n)]
            self.regression_model = None
        else:
            self.regression_model = Mps(field_n, 2, max_bond_dim)
            self.classification_model = None


    def __repr__(self) -> str:
        if self.classification_model is not None:
            return "  DatasetMps(classification)\n{}".format("\n".join(["class: {:3d} --------------".format(i) + model.__repr__() for (i, model) in enumerate(self.classification_model)]))
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
        if self.classification_model is not None:
            l = np.dstack([np.abs(1.0 - m(data2vector(X, x0=self.x0, x1=self.x1))) for m in self.classification_model])
            return np.argmin(l, 2).reshape((X.shape[0], ))
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


    def fit(self, X : np.ndarray, y : np.ndarray, X_test : np.ndarray = None, y_test : np.ndarray = None, learn_rate : float = 0.1, batch_size : int = 10, epochs : int = 50):
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

        if self.classification_model is not None:
            # train the classification models for each class label
            for cls in range(len(self.classification_model)):
                X_cls_train, y_cls_train = data2vector(X, x0=self.x0, x1=self.x1), (y == cls).astype(int)

                X_cls_test, y_cls_test = None, None
                if X_test is not None and y_test is not None:
                    X_cls_test, y_cls_test = data2vector(X_test, x0=self.x0, x1=self.x1), (y_test == cls).astype(int)

                self.classification_model[cls].fit(X_cls_train, y_cls_train, X_cls_test, y_cls_test, learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

        else:
            # train the regression model
            if X_test is not None and y_test is not None:
                X_test = data2vector(X_test, x0=self.x0, x1=self.x1)

            self.regression_model.fit(data2vector(X, x0=self.x0, x1=self.x1), y, X_test, y_test, learn_rate=learn_rate, batch_size=batch_size, epochs=epochs)

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
