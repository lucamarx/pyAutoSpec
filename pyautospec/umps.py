"""
Uniform Matrix Product State
"""
import graphviz
import numpy as np

from typing import List


class UMPS():
    """
    Uniform Matrix Product State (α, A, ω)
    """

    def __init__(self, part_d : int, bond_d : int):
        """
        Parameters
        ----------

        `part_d`: int
        particle dimension

        `bond_d`: int
        bond dimension
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

        `p` : int
        a state

        `w` : float
        the initial weight
        """
        self.alpha[p] = w


    def add_transition(self, p : int, l : str, w : float, q : int):
        """Add transition `p -l-> q` with weight w

        Parameters
        ----------

        `p` : int
        the start state

        `q` : int
        the end state

        `w` : float
        the transition weight
        """
        self.A[p, self.alphabet[l], q] = w


    def set_accepting_state(self, p : int, w : float = 1.0):
        """Set state p as accepting with weight w

        Parameters
        ----------

        `p` : int
        a state

        `w` : float
        the final weight
        """
        self.omega[p] = w


    def evaluate_vword(self, x : np.ndarray) -> float:
        """Evaluate uMPS on a vector word

        Each vector belongs to the particle space

        Parameters
        ----------

        `x`: np.ndarray
        a 2d array with dimensions `(n, part_d)`
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

        Each int in the list refers to a particle dimension

        Parameters
        ----------

        `x`: List[int]
        a list of particle dimensions
        """
        if any([d >= self.part_d for d in x]):
            raise Exception("invalid particle dimension")

        T = self.alpha
        for d in x:
            T = np.einsum("i,ij->j", T, self.A[:,d,:])

        return np.dot(T, self.omega)


    def evaluate_string(self, x : str) -> float:
        """Evaluate uMPS on a string

        Each letter refers to a particle dimension

        Parameters
        ----------

        `x`: str
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
        """State diagram"""

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
