"""
UMps based multi-dimensional functions
"""
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


    def __call__(self, *args) -> float:
        """Evaluate"""
        return self.umps(self.encoder.encode(*args))


    def learn(self, f : Callable[[Tuple], float], learn_resolution : int):
        """Learn a function

        """
        self.f = f
        self.umps.fit(lambda x: f(*self.encoder.decode(x)), learn_resolution)
