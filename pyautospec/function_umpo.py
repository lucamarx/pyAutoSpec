"""
UMpo based multi-dimensional relations
"""
from __future__ import annotations

from typing import Tuple, Optional, Callable

from .umpo import UMPO
from .encoder import VectorEncoder
from .function_umps import FunctionUMps


class FunctionUMpo():
    """
    UMpo based multi-dimensional relation
    """

    def __init__(self, limits : Tuple[float, float], encoding_length : Optional[int] = 12, max_bond_dim : Optional[int] = 20):
        """Create a multi dimensional relation

        Paramaters
        ----------

        limits : Tuple[float,float]
        The limits of each vector dimension

        encoding_length : int, optional
        Default encoding length

        max_bond_dim : int, optional
        The maximum number of states

        """
        self.encoder = VectorEncoder([limits], encoding_length)
        self.umpo = UMPO(self.encoder.part_d, self.encoder.part_d, max_bond_dim)
        self.r = None

    def __repr__(self):
        return f"""
  {" x ".join([f"[{x0:.2f},{x1:.2f})" for (x0, x1) in self.encoder.limits*2])}
  {self.umpo.__repr__()}
        """


    def __call__(self, other : FunctionUMps) -> FunctionUMps:
        """Evaluate over another function

        """
        if self.encoder != other.encoder:
            raise Exception("must be defined on the same domain")

        V = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)
        V.umps = self.umpo(other.umps)

        return V


    def fit(self, r : Callable[[float, float], float], learn_resolution : int, n_states : Optional[int] = None):
        """Learn a weighted relation

        Parameters
        ----------

        r : Callable[[Tuple[float,float]], float]
        The weighted relation to learn

        learn_resolution : int
        The maximum length of words included in the basis used to estimate
        Hankel blocks

        n_states : int, optional
        Truncate the uMps to the specified number of states

        """
        self.umpo.fit(lambda x,y: r(self.encoder.decode(x)[0], self.encoder.decode(y)[0]), learn_resolution, n_states)
        self.r = r
