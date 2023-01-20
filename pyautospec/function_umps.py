"""
UMps based multi-dimensional functions
"""
from __future__ import annotations

import numpy as np

from typing import List, Tuple, Optional, Callable

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

        ∫ `self(x) * other(x)` dx

        over the whole domain

        Paramaters
        ----------

        other : FunctionUMps
        Another function defined on the same domain

        """
        if self.encoder.limits != other.encoder.limits:
            raise Exception("the two functions must be defined on the same domain")

        dx = self.encoder.resolution()[0]

        return dx*self.umps.scalar(other.umps, self.encoder.encoding_length)


    def __call__(self, *args) -> float:
        """Evaluate"""
        return self.umps(self.encoder.encode(*args))


    def fit(self, f : Callable[[Tuple], float], learn_resolution : int, n_states : Optional[int] = None):
        """Learn a function

        """
        self.f = f
        self.umps.fit(lambda x: f(*self.encoder.decode(x)), learn_resolution, n_states)


    def integral(self) -> float:
        """Compute the integral

        ∫ `self(x)**2` dx

        over the whole domain

        """
        if len(self.encoder.limits) != 1:
            # TODO: evaluate over n-dim functions
            raise Exception("integrals can be evaluated for one dim functions only")

        dx = self.encoder.resolution()[0]

        _, I = self.umps.integral(self.encoder.encoding_length)

        i0 = np.einsum("i,p,ipj,j", I, np.array([1,0]), self.umps.A, self.umps.omega)
        i1 = np.einsum("i,p,ipj,j", I, np.array([0,1]), self.umps.A, self.umps.omega)

        return dx*(i1+i0)/2
