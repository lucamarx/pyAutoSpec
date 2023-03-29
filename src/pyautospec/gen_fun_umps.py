"""
UMps based generating functions
"""
from typing import Optional, Callable

from .umps import UMPS
from .encoder import IntegerEncoder


class GenFunctionUMps:
    """UMps based generating functions

    """

    def __init__(self, max_bond_dim : Optional[int] = 20, base : Optional[int] = 2):
        """Create an ordinary generating function

        Parameters
        ----------

        max_bond_dim : int, optional
        The maximum number of states

        """
        self.encoder = IntegerEncoder(base)
        self.umps = UMPS(base, max_bond_dim)
        self.f = None


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


    def __getitem__(self, n : int) -> int:
        """Evaluate n-th term of the power serie

        """
        return round(self.umps(self.encoder.encode(n)))


    def fit(self, f : Callable[[int], int], learn_resolution : int, n_states : Optional[int] = None):
        """Learn a recursive function

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
        self.umps.fit(lambda n: f(self.encoder.decode(n)), learn_resolution, n_states)
        self.f = f
