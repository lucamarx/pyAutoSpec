"""
UMps based multi-dimensional functions
"""
from __future__ import annotations

import numpy as np

from typing import List, Tuple, Optional, Callable, Union

from .umps import UMPS
from .encoder import VectorEncoder


class FunctionUMps():
    """
    UMps based multi-dimensional function
    """

    def __init__(self, limits : List[Tuple[float, float]], encoding_length : Optional[int] = 12, max_bond_dim : Optional[int] = 20):
        """Create a multi dimensional real function

        Paramaters
        ----------

        limits : List[Tuple[float,float]]
        The limits of each vector dimension

        encoding_length : int, optional
        Default encoding length

        max_bond_dim : int, optional
        The maximum number of states

        """
        self.encoder = VectorEncoder(limits, encoding_length)
        self.umps = UMPS(self.encoder.part_d, max_bond_dim)
        self.f = None


    def __repr__(self):
        return f"""
  {" x ".join([f"[{x0:.2f},{x1:.2f})" for (x0, x1) in self.encoder.limits])} → R
  {self.umps.__repr__()}
        """


    def scalar(self, other : FunctionUMps) -> float:
        """Compute the integral

        ⌠
        ⎮ `self(x)·other(x)` dx
        ⌡

        over the whole domain

        Paramaters
        ----------

        other : FunctionUMps
        Another function defined on the same domain

        Returns
        -------

        The value of the integral

        """
        if self.encoder.limits != other.encoder.limits:
            raise Exception("the two functions must be defined on the same domain")

        dx = self.encoder.resolution()[0]

        return dx*self.umps.scalar(other.umps, self.encoder.encoding_length)


    def __call__(self, *args) -> float:
        """Evaluate"""
        return self.umps(self.encoder.encode(*args))


    def __mul__(self, other : Union[int, float, FunctionUMps]) -> FunctionUMps:
        """Product with a scalar or another function

        """
        if isinstance(other, int) or isinstance(other, float):

            P = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)

            P.umps = other * self.umps

            if self.f is not None:
                P.f = lambda *args: other * self.f(*args)

            return P

        if isinstance(other, FunctionUMps):
            if self.encoder != other.encoder:
                raise Exception("must have the same domain/encoding")

            P = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)

            P.umps = other.umps * self.umps

            if self.f is not None and other.f is not None:
                P.f = lambda *args: other.f(*args) * self.f(*args)

            return P

        raise Exception(f"invalid type for function uMps product: {other.__class__.__name__}")


    def __rmul__(self, other : Union[int, float, FunctionUMps]) -> FunctionUMps:
        """Product with a scalar or another function

        """
        return self.__mul__(other)


    def __add__(self, other : FunctionUMps) -> FunctionUMps:
        """Combine two functions by taking their uMps tensor sum

        """
        if self.encoder != other.encoder:
            raise Exception("must have the same domain/encoding")

        S = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)

        S.umps = self.umps + other.umps

        if self.f is not None and other.f is not None:
            S.f = lambda *args: self.f(args) + other.f(args)

        return S


    def __or__(self, other : FunctionUMps) -> FunctionUMps:
        """Combine two functions by taking the concatenation of their uMps

        """
        if self.encoder != other.encoder:
            raise Exception("must have the same domain/encoding")

        C = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)

        C.umps = self.umps | other.umps

        return C


    def __matmul__(self, other : FunctionUMps) -> FunctionUMps:
        """Tensor product over the particle dimension

        """
        if self.encoder != other.encoder:
            raise Exception("must have the same domain/encoding")

        P = FunctionUMps(self.encoder.limits + other.encoder.limits, self.encoder.encoding_length)

        P.umps = self.umps @ other.umps

        return P


    def __mod__(self, other : FunctionUMps) -> FunctionUMps:
        """Tensor sum over the particle dimension of two uMps

        """
        if self.encoder != other.encoder:
            raise Exception("must have the same domain/encoding")

        S = FunctionUMps(self.encoder.limits + other.encoder.limits, self.encoder.encoding_length)

        S.umps = self.umps % other.umps

        return S


    def eval_super(self, args : List[Tuple[float, Tuple]]) -> float:
        """Evaluate function over linear superposition of arguments

        Parameters
        ----------

        args : List[Tuple[float, Tuple[float]]]
        A list `[(a, (u1,u2,...)), (b, (v1,v2,...)), ...]` of
        weighted arguments

        Returns
        -------

        The value of the uMps over the linear combination of encoded arguments

        """
        return self.umps(np.sum([a * self.encoder.one_hot(self.encoder.encode(*x))[0] for a, x in args], axis=0))


    def fit(self, f : Callable[[Tuple], float], learn_resolution : int, n_states : Optional[int] = None):
        """Learn a function

        Parameters
        ----------

        f : Callable[[Tuple], float]
        The function to learn

        learn_resolution : int
        The maximum length of words included in the basis used to estimate
        Hankel blocks

        n_states : int, optional
        Truncate the uMps to the specified number of states

        """
        self.umps.fit(lambda x: f(*self.encoder.decode(x)), learn_resolution, n_states)
        self.f = f


    def integral(self) -> float:
        """Compute the integral

        ⌠
        ⎮ `self(x)` dx
        ⌡

        over the whole domain

        Returns
        -------

        The value of the integral

        """
        if len(self.encoder.limits) != 1:
            # TODO: evaluate over n-dim functions
            raise Exception("integrals can be evaluated for one dim functions only")

        dx = self.encoder.resolution()[0]

        _, I = self.umps.integral(self.encoder.encoding_length)

        i0 = np.einsum("i,p,ipj,j", I, np.array([1,0]), self.umps.A, self.umps.omega)
        i1 = np.einsum("i,p,ipj,j", I, np.array([0,1]), self.umps.A, self.umps.omega)

        return dx*(i1+i0)/2


    def gradient(self, d : Optional[int] = 0) -> FunctionUMps:
        """Compute gradient along dimension

        Parameters
        ----------

        d : int, optional
        The dimension to take the gradient along

        Returns
        -------

        `∂_d self`

        An uMps computing the gradient

        """
        if d < 0 or d > self.encoder.part_d:
            raise Exception("invalid direction")

        G = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)

        x0, x1 = self.encoder.limits[d]

        G.umps.alpha = self.umps.alpha / (x1 - x0)
        G.umps.A     = 2 * self.umps.A
        G.umps.omega = 2 * np.dot(self.umps.A[:,1,:] - self.umps.A[:,0,:], self.umps.omega)

        # TODO: extend to multi-dimensional case

        return G
