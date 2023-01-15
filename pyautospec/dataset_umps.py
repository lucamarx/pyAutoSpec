"""
UMps based data modeling
"""
from __future__ import annotations

import numpy as np

from typing import List, Tuple, Optional, Callable
from tqdm.auto import tqdm

from .umps import UMPS
from .encoder import VectorEncoder


class DatasetUMps():
    """
    UMps based dataset modeling
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
  (({",".join([f"[{x0:.2f},{x1:.2f})" for (x0, x1) in self.encoder.limits])}), y)
  {self.umps.__repr__()}
        """

    def __call__(self, *args) -> float:
        """Evaluate"""
        return self.umps(self.encoder.encode(*args))


    def fit(self, X : np.ndarray, y : np.ndarray):
        """Fit model to data

        Parameters
        ----------

        X : ndarray

        y: np.ndarray

        """
        pass
