"""
Wfa based function compression algorithm
"""
import jax.numpy as jnp

from .wfa import Wfa
from .plots import function_wfa_comparison_chart
from .ps_basis import KBasis
from .spectral_learning import hankel_blocks_for_function


def word2real(s : str, x0 : float = 0.0, x1 : float = 1.0):
    """
    Convert the binary representation s of xϵ[x0,x1) into the number itself
    """
    s = "0" + s
    return x0 + sum([int(s[i]) * 2**(-i) for i in range(0,len(s))]) * (x1-x0)


def real2word(r : float, l : int = 8, x0 : float = 0.0, x1 : float = 1.0):
    """
    Convert a real number xϵ[x0,x1) into its binary representation (with
    maximum length l)
    """
    if r < x0 or r >= x1:
        raise Exception("out of bounds")

    r = (r - x0) / (x1 - x0)
    w = ""
    for _ in range(0,l+1):
        d = 1 if r >= 1 else 0
        w += str(d)
        r = (r-d)*2
    return w[1:]


def wfa_derivative(wfa, x0, x1):
    """
    Compute wfa's derivative
    """
    wfa_d = Wfa(["0", "1"], len(wfa))

    wfa_d.alpha = wfa.alpha / (x1 - x0)
    wfa_d.A     = 2 * wfa.A
    wfa_d.omega = 2 * jnp.dot(wfa.A[:,1,:] - wfa.A[:,0,:], wfa.omega)

    return wfa_d


class FunctionWfa():
    """
    Wfa based real function model
    """

    def __init__(self):
        """
        Intialize learn a model of a real function f: [x0,x1) → R
        """
        self.f, self.x0, self.x1 = None, None, None

        self.model = Wfa(["0", "1"], 2)


    def __repr__(self):
        if self.f is None:
            return "  FunctionWfa(N={}) <?>: [<?>,<?>] → R\n{}".format(len(self.model), self.model.__repr__())
        else:
            return "  FunctionWfa(N={}) {}: [{:.2f},{:.2f}] → R\n{}".format(len(self.model), self.f.__repr__(), self.x0, self.x1, self.model.__repr__())


    def __call__(self, x : float, resolution : int = 12):
        """
        Evaluate learned function at x

        Parameters:
        -----------

        x : float
        a point in [x0,x1)

        Returns:
        --------

        the value of the function at x
        """
        return self.model(real2word(x, l=resolution, x0=self.x0, x1=self.x1))


    def prime(self, x : float, resolution : int = 12):
        """
        Evaluate derivative at x

        Parameters:
        -----------

        x : float
        a point in [x0,x1)

        Returns:
        --------

        the value of the derivative of the function at x
        """
        return self.deriv(real2word(x, l=resolution, x0=self.x0, x1=self.x1))


    def comparison_chart(self, n_points : int = 50, resolution : int = 12, plot_derivative : bool = False):
        """
        Compare the two functions

        Parameters:
        -----------

        n_points : int
        the number of points in the plot

        resolution : int
        the word length used to encode x's values

        plot_derivative : bool
        whether to plot the derivative
        """
        function_wfa_comparison_chart(self, n_points, resolution, plot_derivative)


    def fit(self, f, x0 : float = 0.0, x1 : float = 1.0, learn_resolution : int = 3):
        """
        Fit the model to the function f defined on the interval [x0,x1)

        Parameters:
        -----------

        f : function
        the function to be fitted

        x0 : float
        x1 : float
        the interval the function is defined on

        learn_resolution : int
        the length of the basis used to construct the Hankel blocks

        Returns:
        --------

        The object itself
        """
        if x1 <= x0:
            raise Exception("x0 must be less than x1")

        self.f = f
        self.x0 = x0
        self.x1 = x1

        basis = KBasis(["0", "1"], learn_resolution)

        hp, H, Hs, hs = hankel_blocks_for_function(lambda w: f(word2real(w, x0=x0, x1=x1)), basis)

        self.model.fit(hp, H, Hs, hs)
        self.deriv = wfa_derivative(self.model, x0, x1)

        return self
