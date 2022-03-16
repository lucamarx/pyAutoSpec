"""
Wfa based function compression algorithm
"""
import itertools
from .spectral_learning import SpectralLearning


def decode(w : str, x0 = 0.0, x1 = 1.0):
    """
    Decode word → x coordinate
    """
    if w == "":
        return x0 + (x1-x0) / 2

    if w[0] == "a":
        x1 = x0 + (x1-x0)/2
    elif w[0] == "b":
        x0 = x1 - (x1-x0)/2

    return decode(w[1:], x0, x1)


class FunctionWfa():
    """
    Wfa based function model
    """

    def __init__(self, f, x0 : float = 0.0, x1 : float = 1.0):
        """
        Intialize model
        """
        self.f = f

        self.x0 = x0
        self.x1 = x1

        self.model = None


    def decode(self, w : str):
        """
        Decode a word into x
        """
        return decode(w, x0=self.x0, x1=self.x1)


    def learn(self, resolution=4):
        """
        Learn model from the function f
        """
        learn = SpectralLearning(["a", "b"], resolution)
        self.model = learn.learn(lambda w: self.f(decode(w, x0=self.x0, x1=self.x1)))


    def reconstruct(self, resolution=4):
        """
        Reconstruct the function from the model
        """
        points, original, reconstr = [], [], []

        for w in [''.join(w) for w in itertools.product(*([["a", "b"]] * resolution))]:
            x = decode(w, x0=self.x0, x1=self.x1)

            points.append(x)
            original.append(self.f(x))
            reconstr.append(self.model.evaluate(w))

        return points, original, reconstr
