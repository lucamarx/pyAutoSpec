"""
Uniform Matrix Product State
"""
import graphviz
import numpy as np

from typing import List


def pseudo_inverse(M : np.ndarray) -> np.ndarray:
    """Moore-Penrose pseudo inverse

    The pseudo inverse of a matrix M is

    M^†^ = V Σ^†^ U*

    where V,Σ,U are M's SVD factors:

    M = U Σ V*

    see [Moore–Penrose inverse](https://en.wikipedia.org/wiki/Moore–Penrose_inverse)

    """
    U, S, Vt = np.linalg.svd(M, full_matrices=True, compute_uv=True)

    Sinv = np.zeros((U.shape[1], Vt.shape[0]), dtype=np.float32)
    diag = np.diag_indices(S.shape[0])
    Sinv[diag] = 1/S
    Sinv = np.transpose(Sinv)

    return np.dot(np.dot(np.transpose(Vt), Sinv), np.transpose(U))


class UMPS():
    """
    Uniform Matrix Product State (α, A, ω)
    """

    def __init__(self, part_d : int, bond_d : int):
        """Create a uMPS

        Parameters
        ----------

        part_d : int
        Particle dimension

        bond_d : int
        Bond dimension

        """
        self.part_d = part_d
        self.bond_d = bond_d

        self.alpha = np.zeros((bond_d,), dtype=np.float32)
        self.A = np.zeros((bond_d, part_d, bond_d), dtype=np.float32)
        self.omega = np.zeros((bond_d,), dtype=np.float32)

        self.alphabet = {chr(97+i):i for i in range(part_d)}


    def __repr__(self) -> str:
        return """
  ╭───┐       ╭───┐       ╭───┐
  │ α ├─ ... ─┤ A ├─ ... ─┤ ω │
  └───┘       └─┬─┘       └───┘

  particle dim: {:3d}
      bond dim: {:3d}
        """.format(self.part_d, self.bond_d)


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
        """Evaluate uMPS on a vector word

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


    def evaluate_list(self, x : List[int]) -> float:
        """Evaluate uMPS on a list of integers

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


    def evaluate_string(self, x : str) -> float:
        """Evaluate uMPS on a string

        Parameters
        ----------

        x : str
        A word where each letter refers to a particle dimension

        Returns
        -------

        float
        The value of the uMPS over the input `x`

        """
        return self.evaluate_list([self.alphabet[c] for c in x])


    def __call__(self, x) -> float:
        """Evaluate uMPS"""

        if isinstance(x, np.ndarray):
            return self.evaluate_vword(x)

        if isinstance(x, list):
            return self.evaluate_list(x)

        if isinstance(x, str):
            return self.evaluate_string(x)

        raise Exception("invalid type")


    def diagram(self, title : str = "uMPS", epsilon : float = 1e-8):
        """Draw state diagram"""

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


    def spectral_learning(self, hp : np.ndarray, H : np.ndarray, Hs : np.ndarray, hs : np.ndarray, n_states : int = None):
        """Spectral learning algorithm

        Perform spectral learning of Hankel blocks truncating expansion to
        `n_states` (if specified)

        Parameters
        -----------

        hp : np.ndarray

        H : np.ndarray

        Hs : np.ndarray

        hs : np.ndarray

        n_states : int

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

        # reset part/bond dimensions, alphabet
        self.bond_d = self.A.shape[0]
        self.part_d = self.A.shape[1]

        self.alphabet = {chr(97+i):i for i in range(self.part_d)}
