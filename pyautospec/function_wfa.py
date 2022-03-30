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


def word2real(s : str, x0 : float = 0.0, x1 : float = 1.0):
    """
    Convert the binary representation s of xϵ[x0,x1] into the number itself
    """
    s = "0" + s
    return x0 + sum([int(s[i]) * 2**(-i) for i in range(0,len(s))]) * (x1-x0)


def real2word(r : float, l : int = 8, x0 : float = 0.0, x1 : float = 1.0):
    """
    Convert a real number xϵ[x0,x1] into its binary representation (with
    maximum length l)
    """
    if r < x0 or r > x1:
        raise Exception("out of bounds")

    r = (r - x0) / (x1 - x0)
    w = ""
    for _ in range(0,l+1):
        d = 1 if r >= 1 else 0
        w += str(d)
        r = (r-d)*2
    return w[1:]


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
