"""
UMps based generating functions
"""
import numpy as np

from math import log2
from typing import List, Tuple, Optional, Callable, Union

from .umps import UMPS
from .encoder import IntegerEncoder


class GenFunctionUMps:
    """UMps based generating functions

    """

    def __init__(self, max_bond_dim : Optional[int] = 20):
        """Create a generating function

        Parameters
        ----------

        max_bond_dim : int, optional
        The maximum number of states

        """
        self.encoder = IntegerEncoder()
        self.umps = UMPS(2, max_bond_dim)


    def __repr__(self):
        return f"""
         n
   Σ a  x
      n
  {self.umps.__repr__()}
        """


    def __call__(self, x : float) -> float:
        """Evaluate generating function at x

        f(x) = Σ a_n x^n

        """
        # TODO: make it more efficient
        return sum([self[n] * x**n for n in range(100)])


    def __getitem__(self, n : int) -> float:
        """Evaluate n-th term of the power serie

        """
        return self.umps(self.encoder.encode(n))
