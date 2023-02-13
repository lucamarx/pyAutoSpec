"""
UMpo based multi-dimensional relations
"""
from __future__ import annotations

from typing import Tuple, Optional, Callable, Union

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
  {" x ".join([f"[{x0:.2f},{x1:.2f})" for (x0, x1) in self.encoder.limits*2])} → R
  {self.umpo.__repr__()}
        """


    def __call__(self, other : Union[FunctionUMps, Tuple[float,float]]) -> FunctionUMps:
        """Evaluate operator over another uMps or on a pair of values

        """
        if isinstance(other, FunctionUMps):
            if self.encoder != other.encoder:
                raise Exception("must be defined on the same domain")

            V = FunctionUMps(self.encoder.limits, self.encoder.encoding_length)
            V.umps = self.umpo(other.umps)

            return V

        if isinstance(other, tuple):
            return self.umpo((self.encoder.encode(other[0]), self.encoder.encode(other[1])))


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
