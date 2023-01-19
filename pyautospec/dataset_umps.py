"""
UMps based data modeling
"""
from __future__ import annotations

import numpy as np

from typing import Dict, List, Tuple, Optional
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



    def _c_basis(self, X : np.ndarray, Xs : np.ndarray) -> Tuple[Dict, Dict]:
        """Take prefixes/suffixes from a list of words

        Parameters
        ----------

        X : np.ndarray
        An encoded dataset

        Returns
        -------

        `(prefixes : Dict, suffixes : Dict)`

        The dictionaries of the prefixes/suffixes found in the dataset

        """
        prefixes, suffixes = {}, {}
        last_prefix, last_suffix = 0, 0
        for x in X:
            for i in range(len(x)):
                p, q = tuple(x[:i]), tuple(x[i:])

                if prefixes.get(p) is None:
                    prefixes[p] = last_prefix
                    last_prefix += 1

                if suffixes.get(q) is None:
                    suffixes[q] = last_suffix
                    last_suffix += 1

        for x in Xs:
            if len(x) < 2:
                continue

            for i in range(1,len(x)):
                p, q = tuple(x[:i-1]), tuple(x[i:])

                if prefixes.get(p) is None:
                    prefixes[p] = last_prefix
                    last_prefix += 1

                if suffixes.get(q) is None:
                    suffixes[q] = last_suffix
                    last_suffix += 1

        return prefixes, suffixes


    def _hankel_blocks(self, X : np.ndarray, Xs : np.ndarray, y : np.ndarray, prefixes : Dict, suffixes : Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Hankel blocks for function over a basis

        Parameters
        ----------

        X : np.ndarray

        Xs : np.ndarray

        y : np.ndarray

        prefixes : Dict

        suffixes : Dict

        Returns
        -------

        `(hp, hs, H, Hs)`

        The estimated Hankel blocks for `f` over the basis

        """

        hp = np.zeros((len(prefixes),), dtype=np.float32)
        H  = np.zeros((len(prefixes), len(suffixes)), dtype=np.float32)
        Hs = np.zeros((len(prefixes), self.encoder.part_d, len(suffixes)), dtype=np.float32)
        hs = np.zeros((len(suffixes),), dtype=np.float32)

        # compute Hankel blocks
        for n in tqdm(range(len(X))):
            x = X[n]
            t = tuple(x)

            if prefixes.get(t) is not None:
                hp[prefixes[t]] = y[n]

            if suffixes.get(t) is not None:
                hs[suffixes[t]] = y[n]

            for i in range(len(x)):
                p, q = tuple(x[:i]), tuple(x[i:])
                H[prefixes[p], suffixes[q]] = y[n]

        for n in tqdm(range(len(Xs))):
            x = Xs[n]
            t = tuple(x)

            if prefixes.get(t) is not None:
                hp[prefixes[t]] = y[n]

            if suffixes.get(t) is not None:
                hs[suffixes[t]] = y[n]

            if len(x) < 2:
                continue

            for i in range(1, len(x)):
                p, q = tuple(x[:i-1]), tuple(x[i:])
                Hs[prefixes[p], x[i-1], suffixes[q]] = y[n]

        return hp, hs, H, Hs


    def fit(self, X : np.ndarray, y : np.ndarray, learn_resolution : int, n_states : Optional[int] = None):
        """Fit model to data

        Parameters
        ----------

        X : ndarray

        y: np.ndarray

        learn_resolution : int

        n_states : int, optional

        """
        # encode X as v-words
        X_enc = VectorEncoder(self.encoder.limits, learn_resolution).encode_array(X)

        # encode X as one letter longer v-words
        Xs_enc = VectorEncoder(self.encoder.limits, learn_resolution+1).encode_array(X)

        # compute basis from words
        prefixes, suffixes = self._c_basis(X_enc, Xs_enc)

        # estimate Hankel blocks
        hp, hs, H, Hs = self._hankel_blocks(X_enc, Xs_enc, y, prefixes, suffixes)

        # COMPLETE Hankel blocks?

        # perform spectral learning from model
        self.umps._spectral_learning(hp, hs, H, Hs, n_states)
