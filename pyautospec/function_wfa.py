"""
Wfa based function compression algorithm
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
    Wfa based real function model
    """

    def __init__(self, f, x0 : float = 0.0, x1 : float = 1.0, learn_resolution : int = 3):
        """
        Intialize learn a model of a real function f: [x0,x1] → R
        """
        self.f = f

        self.x0 = x0
        self.x1 = x1

        self.splrn = SpectralLearning(["0", "1"], learn_resolution)
        self.model = self.splrn.learn(lambda w: self.f(word2real(w, x0=x0, x1=x1)))


    def __repr__(self):
        return "WFA {}: [{:.2f},{:.2f}] → R".format(self.f.__repr__(), self.x0, self.x1)


    def w2r(self, w : str):
        """
        Convert binary representation to real number
        """
        return word2real(w, x0=self.x0, x1=self.x1)


    def r2w(self, r : float, l : int = 6):
        """
        Convert real number to its binary representation
        """
        return real2word(r, x0=self.x0, x1=self.x1, l=l)


    def __call__(self, x : float, resolution : int = 64):
        """
        Evaluate learned function at x
        """
        return self.model(real2word(x, l=resolution, x0=self.x0, x1=self.x1))


    def comparison_chart(self, n_points : int = 50):
        """
        Compare the two functions
        """
        xs = jnp.linspace(self.x0, self.x1, num = n_points)

        v0 = jnp.array([self.f(x) for x in xs])
        v1 = jnp.array([self(x) for x in xs])

        error = jnp.abs(v1 - v0)

        plt.title("{} reconstruction error: avg={:.2f} max={:.2f} ".format(self.f.__repr__(), jnp.average(error), jnp.max(error)))

        plt.plot(xs, v0, label="original")
        plt.plot(xs, v1, label="reconstructed")
        plt.legend()
