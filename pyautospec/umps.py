"""
Uniform Matrix Product State
"""
from __future__ import annotations

import graphviz
import itertools
import numpy as np

from typing import List, Tuple, Optional, Callable
from tqdm.auto import tqdm


def pseudo_inverse(M : np.ndarray) -> np.ndarray:
    """Moore-Penrose pseudo inverse

    The pseudo inverse of a matrix M is

    M^†^ = V Σ^†^ U*

    where V,Σ,U are M's SVD factors:

    M = U Σ V*

    see [Moore–Penrose inverse](https://en.wikipedia.org/wiki/Moore–Penrose_inverse)

    Parameters
    ----------

    M : np.ndarray

    Returns
    -------

    np.ndarray

    The pseudo inverse

    """
    U, S, Vt = np.linalg.svd(M, full_matrices=True, compute_uv=True)

    Sinv = np.zeros((U.shape[1], Vt.shape[0]), dtype=np.float32)
    diag = np.diag_indices(S.shape[0])
    Sinv[diag] = 1/S
    Sinv = np.transpose(Sinv)

    return np.dot(np.dot(np.transpose(Vt), Sinv), np.transpose(U))


class UMPS():
    """Uniform Matrix Product State (α, A, ω)

    Here we we consider the particle dimensions and the alphabet as the same
    thing

    """

    def __init__(self, part_d : int, bond_d : int, alphabet : Optional[List[str]] = None):
        """Create a uMPS

        Parameters
        ----------

        part_d : int
        Particle dimension (or equivalently alphabet size)

        bond_d : int
        Bond dimension

        """

        self.alpha = np.zeros((bond_d,), dtype=np.float32)
        self.A = np.zeros((bond_d, part_d, bond_d), dtype=np.float32)
        self.omega = np.zeros((bond_d,), dtype=np.float32)

        self.singular_values = None

        if alphabet is None:
            self.alphabet = {chr(97+i):i for i in range(part_d)}
        else:
            if len(alphabet) != part_d:
                raise Exception("there should be one letter for each particle dimension")
            self.alphabet = {alphabet[i]:i for i in range(len(alphabet))}


    @property
    def part_d(self) -> int:
        return self.A.shape[1]


    @property
    def bond_d(self) -> int:
        return self.A.shape[0]


    def __repr__(self) -> str:

        S = f"{self.entropy():.2f} (bits)" if self.singular_values is not None else "-"

        return f"""
  ╭───┐     ╭───┐     ╭───┐
  │ α ├─...─┤ A ├─...─┤ ω │
  └───┘     └─┬─┘     └───┘

  particle dim: {self.part_d:3d}
      bond dim: {self.bond_d:3d}
       entropy: {S}
        """


    def set_starting_state(self, p : int, w : float = 1.0):
        """Set state p as starting with weight w

        Parameters
        ----------

        p : int
        The initial state

        w : float, default=1
        The initial weight

        """
        self.alpha[p] = w


    def add_transition(self, p : int, l : str, w : float, q : int):
        """Add transition `p -l-> q` with weight w

        Parameters
        ----------

        p : int
        The transition start state

        l : str
        The transition letter

        q : int
        The transition final state

        w : float
        The transition weight

        """
        self.A[p, self.alphabet[l], q] = w


    def set_accepting_state(self, p : int, w : float = 1.0):
        """Set state p as accepting with weight w

        Parameters
        ----------

        p : int
        The final state

        w : float
        The final state weight
        """
        self.omega[p] = w


    def evaluate_vword(self, x : np.ndarray) -> float:
        """Evaluate uMPS on a v-word (vector word)

        Each vector belongs to the particle space

        Parameters
        ----------

        x: np.ndarray
        An array with dimensions `(n, part_d)` of vectors each
        living in the particle space

        Returns
        -------

        float
        The value of the uMPS over the input `x`

        """
        if len(x.shape) != 2:
            raise Exception("invalid v-word shape")

        if x.shape[1] != self.part_d:
            raise Exception("invalid particle dimension")

        T = self.alpha
        for n in range(x.shape[0]):
            T = np.einsum("i,ipj,p->j", T, self.A, x[n])

        return np.dot(T, self.omega)


    def evaluate_lword(self, x : List[int]) -> float:
        """Evaluate uMPS on a l-word (list of particle dimension)

        Parameters
        ----------

        x : List[int]
        A list where each element refers to a particle dimension

        Returns
        -------

        float
        The value of the uMPS over the input `x`

        """
        if any([d >= self.part_d for d in x]):
            raise Exception("invalid particle dimension")

        T = self.alpha
        for d in x:
            T = np.einsum("i,ij->j", T, self.A[:,d,:])

        return np.dot(T, self.omega)


    def evaluate_word(self, x : str) -> float:
        """Evaluate uMPS on a word

        Parameters
        ----------

        x : str
        A word where each letter refers to a particle dimension

        Returns
        -------

        float
        The value of the uMPS over the input `x`

        """
        return self.evaluate_lword([self.alphabet[c] for c in x])


    def __call__(self, x) -> float:
        """Evaluate uMPS"""

        if isinstance(x, np.ndarray):
            return self.evaluate_vword(x)

        if isinstance(x, list):
            return self.evaluate_lword(x)

        if isinstance(x, str):
            return self.evaluate_word(x)

        raise Exception("invalid type")


    def scalar(self, other : UMPS, length : int) -> float:
        """Compute scalar product between two uMPSs

        `    ╭───┐     ╭───┐
        ` α ─┤ A ├─...─┤ A ├─ ω
        `    └─┬─┘     └─┬─┘
        `    ╭─┴─┐     ╭─┴─┐
        ` α'─┤ A'├─...─┤ A'├─ ω'
        `    └───┘     └───┘

        Parameters
        ----------

        other : UMPS
        Another uMPS

        length : int
        Number of tensors to include

        """
        if length < 1:
            raise Exception("length must be at least 1")

        if self.part_d != other.part_d:
            raise Exception("the two uMPSs must have the same particle dimension")

        T = np.einsum("ipk,jpl->ijkl", self.A, other.A)
        S = np.einsum("i,j->ij", self.alpha, other.alpha)
        for _ in range(length):
            S = np.einsum("ij,ijkl->kl", S, T)
        return np.einsum("ij,i,j", S, self.omega, other.omega)


    def diagram(self, title : Optional[str] = "uMPS", epsilon : Optional[float] = 1e-8):
        """Draw state diagram

        Parameters
        ----------

        title : str
        The title

        epsilon : float, optional
        Do not draw transitions with weights less than `epsilon`

        """

        labels = list(self.alphabet.keys())

        dot = graphviz.Digraph(comment=title, graph_attr={"rankdir": "LR",
                                                          "title": title})

        # create states (with weights on accepting states)
        for p in range(self.bond_d):
            if abs(self.omega[p]) > epsilon:
                dot.node(f"{p}", shape="doublecircle")
            else:
                dot.node(f"{p}", shape="circle")

        # add arrow and weight on starting states
        for p in range(self.bond_d):
            if abs(self.alpha[p]) > epsilon:
                dot.node(f"{p}s", label=f"{self.alpha[p]:.2f}", shape="none")
                dot.edge(f"{p}s", f"{p}")

        # add arrow and weights on accepting states
        for p in range(self.bond_d):
            if abs(self.omega[p]) > epsilon:
                dot.node(f"{p}a", label=f"{self.omega[p]:.2f}", shape="none")
                dot.edge(f"{p}", f"{p}a")

        # add weighted transitions
        for p in range(self.bond_d):
            for q in range(self.bond_d):
                for l in range(self.part_d):
                    if abs(self.A[p,l,q]) > epsilon:
                        dot.edge(f"{p}", f"{q}", label=f"{labels[l]}:{self.A[p,l,q]:.2f}")

        return dot


    def _k_basis(self, max_word_length : int):
        """A basis where the prefix/suffix sets are all the words of length less or
        equal than `max_word_length`

        Parameters
        ----------

        max_word_length : int
        Maximum word length

        Yields
        ------

        `(w, i)`

        The word (as a list of particle dimensions) and ints index in the basis
        """
        alphabet = list(range(self.part_d))

        yield(([], 0))

        i = 1
        for l in range(1, max_word_length+1):
            for w in itertools.product(*([alphabet] * l)):
                yield((list(w), i))
                i += 1


    def _hankel_blocks(self, f : Callable[[List[int]], float], basis : List[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate Hankel blocks for function over a basis

        Parameters
        ----------

        f : function
        The function we want to learn

        max_word_length : int
        Maximum word length to be included in basis

        Returns
        -------

        `(h, H, Hs)`

        The estimated Hankel blocks for `f` over the basis

        """
        ps, a = len(basis), self.part_d

        h = np.zeros((ps,), dtype=np.float32)
        H  = np.zeros((ps,ps), dtype=np.float32)
        Hs = np.zeros((ps,a,ps), dtype=np.float32)

        # compute Hankel blocks
        for (u, u_i) in tqdm(basis):
            h[u_i] = f(u)

            for (v, v_i) in basis:
                H[u_i, v_i] = f(u + v)

                for a in range(self.part_d):
                    Hs[u_i, a, v_i] = f(u + [a] + v)

        return h, H, Hs


    def _spectral_learning(self, hp : np.ndarray, hs : np.ndarray, H : np.ndarray, Hs : np.ndarray, n_states : Optional[int] = None):
        """Spectral learning algorithm

        Perform spectral learning of Hankel blocks truncating expansion to
        `n_states` (if specified)

        Parameters
        -----------

        hp : np.ndarray

        hs : np.ndarray

        H : np.ndarray

        Hs : np.ndarray

        n_states : int, optional

        """
        # compute full-rank factorization H = P·S
        U, D, V = np.linalg.svd(H, full_matrices=True, compute_uv=True)

        # truncate expansion
        rank = np.linalg.matrix_rank(H)

        if n_states is not None and n_states < rank:
            rank = n_states

        U, D, V = U[:,0:rank], D[0:rank], V[0:rank,:]
        D_sqrt = np.sqrt(D)

        # TODO: is this the place to make uMPS left/right canonical?
        P = np.einsum("ij,j->ij", U, D_sqrt)
        S = np.einsum("i,ij->ij", D_sqrt, V)

        # compute pseudo inverses
        Pd, Sd = pseudo_inverse(P), pseudo_inverse(S)

        # α = h·S†
        self.alpha = np.dot(hs, Sd)

        # A = P†·Hσ·S†
        self.A = np.einsum("pi,iaj,jq->paq", Pd, Hs, Sd)

        # ω = P†·h
        self.omega = np.dot(Pd, hp)

        # save singular values
        self.singular_values = D


    def fit(self, f : Callable[[List[int]], float], learn_resolution : int, n_states : Optional[int] = None):
        """Learn a function

        f: Σ* → R

        Parameters
        ----------

        f : function
        The function to be learned, it must be defined over list
        in the alphabet with float values

        learn_resolution : int
        The maximum number of letters to use to build Hankel blocks

        n_states : int, optional
        The maximum number of states in the uMPS

        """
        basis = list(self._k_basis(learn_resolution))
        h, H, Hs = self._hankel_blocks(f, basis)
        self._spectral_learning(h, h, H, Hs, n_states)


    def integral(self, length : int) -> Tuple[float, np.ndarray]:
        """Sum over all the words of fixed length

        `    ╭───┐     ╭───┐
        ` α ─┤ A ├─...─┤ A ├─ ω
        `    └─┬─┘     └─┬─┘
        `      1         1

        Parameters
        ----------

        length : int
        Include only words of fixed length

        Returns
        -------

        `(i, I)`

        The uMPS "integral" and the partially evaluated chain

        """
        if length < 1:
            raise Exception("length must be at least 1")

        I, T = self.alpha, np.einsum("p,ipj->ij", np.ones(self.part_d), self.A)
        for _ in range(length):
            I = np.einsum("i,ij->j", I, T)

        return np.dot(I, self.omega), I


    def entropy(self) -> float:
        """Compute entanglement entropy from singular values

        Returns
        -------

            2       2
        -∑ s_i log(s_i)

        """
        if self.singular_values is None:
            raise Exception("no singular values")

        s = np.square(self.singular_values)
        s = s / np.sum(s)
        return -np.sum(s * np.log2(s))
