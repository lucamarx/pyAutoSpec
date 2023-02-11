"""
Uniform Matrix Product Operator
"""
from __future__ import annotations

import numpy as np

from typing import List, Tuple, Optional, Callable, Union

from .umps import UMPS


class UMPO():
    """Uniform Matrix Product Operator (α, A, ω)

    """

    def __init__(self, inp_part_d : int, out_part_d : int, bond_d : int):
        """Create a uMpo

        Parameters
        ----------

        inp_part_d : int
        Input particle dimension

        out_part_d : int
        Output particle dimension

        bond_d : int
        Bond dimension

        """
        self.inp_part_d = inp_part_d
        self.out_part_d = out_part_d

        self.umps = UMPS(inp_part_d * out_part_d, bond_d)

        self.io2alpha = np.array(range(inp_part_d * out_part_d)).reshape(inp_part_d, out_part_d)
        self.alpha2io = {self.io2alpha[i,j]: (i,j) for i in range(inp_part_d) for j in range(out_part_d)}


    def __repr__(self) -> str:

        S = f"{self.umps.entropy():.2f} (bits)" if self.umps.singular_values is not None else "-"

        return f"""
             OUT
  ╭───┐     ╭─┴─┐     ╭───┐
  │ α ├─...─┤ A ├─...─┤ ω │
  └───┘     └─┬─┘     └───┘
             INP

  particle dim:
         input: {self.inp_part_d:3d}
        output: {self.out_part_d:3d}
      bond dim: {self.umps.bond_d:3d}
       entropy: {S}
        """


    def evaluate_umps(self, other : UMPS) -> UMPS:
        """Evaluate uMpo by contracting the input legs with the given uMps

        `  ╭───┐     ╭─┴─┐     ╭───┐
        `  │ α ├─...─┤ A ├─...─┤ ω │
        `  └───┘     └─┬─┘     └───┘
        `  ╭───┐     ╭─┴─┐     ╭───┐
        `  │ β ├─...─┤ B ├─...─┤ ψ │
        `  └───┘     └───┘     └───┘

        Parameters
        ----------

        other : UMPS
        Another uMps to contract with

        Returns
        -------

        The uMps resuting by the contraction

        """
        A_io = self.umps.A[:,self.io2alpha,:]

        P = UMPS(self.out_part_d, self.umps.bond_d * other.bond_d)
        P.alpha = np.einsum("i,k->ik", self.umps.alpha, other.alpha).reshape(P.bond_d)
        P.A     = np.einsum("ipqj,kpl->ikqjl", A_io, other.A).reshape(P.bond_d, self.out_part_d, P.bond_d)
        P.omega = np.einsum("j,l->jl", self.umps.omega, other.omega).reshape(P.bond_d)

        return P


    def __call__(self, x : Union[UMPS, List[int]]) -> Union[UMPS, float]:
        """Evaluate uMpo

        """
        if isinstance(x, UMPS):
            return self.evaluate_umps(x)

        if isinstance(x, list):
            return self.umps(x)

        raise Exception("invalid type")


    def _io_decode(self, w : List[int]) -> Tuple[List[int], List[int]]:
        """Decode a word in the combined alphabet into two words in the
        input/output alphabets

        """
        return ([self.alpha2io[x][0] for x in w], [self.alphabet[x][1] for x in w])


    def fit(self, r : Callable[[Tuple[List[int], List[int]]], float], learn_resolution : int, n_states : Optional[int] = None):
        """Learn a weighted relation

        f: Σ* x Σ* → R

        Parameters
        ----------

        r : Callable
        The relation to be learned, it must be defined over l-words
        in the input/output alphabet with float values

        learn_resolution : int
        The maximum number of letters to use to build Hankel blocks

        n_states : int, optional
        The maximum number of states in the uMpo

        """
        self.umps.fit(lambda x: r(self._io_decode(x)), learn_resolution, n_states)
