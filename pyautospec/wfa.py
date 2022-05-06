"""
Weighted finite automaton implemented as a matrix product state
"""
import graphviz
import numpy as np
import jax.numpy as jnp

from jax import jit
from typing import List
from tqdm.auto import tqdm

from .plots import transition_plot
from .ps_basis import KBasis
from .spectral_learning import spectral_learning, hankel_blocks_for_function


@jit
def evaluate(alpha, A, omega, X):
    y = alpha
    for i in range(0, len(X)):
        y = jnp.dot(y, A[:,X[i],:])
    return jnp.dot(y, omega)


class Wfa:
    """
    Weighted Finite Automaton (α, A, ω) over an alphabet Σ
    """

    def __init__(self, alphabet : List[str], states : int):
        """
        Initialize a WFA with given alphabet and number of states
        """
        self.alphabet_map = {alphabet[i]: i for i in range(0,len(alphabet))}
        self.alpha    = jnp.zeros((states,), dtype=jnp.float32)
        self.A        = jnp.zeros((states, len(alphabet), states), dtype=jnp.float32)
        self.omega    = jnp.zeros((states,), dtype=jnp.float32)


    def __repr__(self):
        return "WFA(states={})".format(len(self))


    def __len__(self):
        """
        The length is just the number of states
        """
        return self.A.shape[0]


    def set_starting_state(self, p : int, w : float = 1):
        """
        Set state p as starting with weight w

        Parameters:
        -----------

        p : int
        a state

        w : float
        the initial weight
        """
        self.alpha = self.alpha.at[p].set(w)


    def add_transition(self, p : int, l : str, w : float, q : int):
        """
        Add transition p -l-> q with weight w

        Parameters:
        -----------

        p : int
        the start state

        p : int
        the end state

        w : float
        the transition weight
        """
        self.A = self.A.at[p, self.alphabet_map[l], q].set(w)


    def set_accepting_state(self, p : int, w : float = 1.0):
        """
        Set state p as accepting with weight w

        Parameters:
        -----------

        p : int
        a state

        w : float
        the final weight
        """
        self.omega = self.omega.at[p].set(w)


    def __call__(self, X : List[str]) -> float:
        """
        Evaluate WFA over word X
        """
        return evaluate(self.alpha, self.A, self.omega, [self.alphabet_map[a] for a in X]).item()


    def fit(self, hp : np.ndarray, H : np.ndarray, Hs : np.ndarray, hs : np.ndarray):
        """
        Learn WFA from Hankel blocks using spectral learning

        Parameters:
        -----------

        hp : np.ndarray

        H  : np.ndarray

        Hs : np.ndarray

        hs : np.ndarray
        """
        self.alpha, self.A, self.omega = spectral_learning(hp, H, Hs, hs)

        return self


    def diagram(self, title : str = "WFA"):
        """
        Make diagram
        """
        labels = {i : l for (l, i) in self.alphabet_map.items()}

        dot = graphviz.Digraph(comment=title, graph_attr={"rankdir": "LR", "title": title})

        # create states (with weights on accepting states)
        for p in range(0, self.A.shape[0]):
            if self.omega[p] != 0:
                dot.node(str(p), shape="doublecircle")
            else:
                dot.node(str(p), shape="circle")

        # add arrow and weight on starting states
        for p in range(0, self.alpha.shape[0]):
            if self.alpha[p] != 0:
                dot.node("{}s".format(p), label="{:.1f}".format(self.alpha[p]), shape="none")
                dot.edge("{}s".format(p), str(p))

        # add arrow and weights on accepting states
        for p in range(0, self.omega.shape[0]):
            if self.omega[p] != 0:
                dot.node("{}a".format(p), label="{:.1f}".format(self.omega[p]), shape="none")
                dot.edge(str(p), "{}a".format(p))

        # add transitions
        for p in range(0, self.A.shape[0]):
            for q in range(0, self.A.shape[2]):
                for l in range(0, self.A.shape[1]):
                    if self.A[p,l,q] > 0:
                        dot.edge(str(p), str(q), label="{}:{:.1f}".format(labels.get(l, str(l)), self.A[p,l,q]))

        return dot


    def transition_plot(self, X : List[str], threshold : float = 1e-4):
        """
        Plot weight contributions
        """
        transition_plot(self, X, threshold)


class SpectralLearning():
    """
    Spectral learning algorithm:

    - generate a base made of words with maximum length
    - build the Hankel matrix for the function f
    - perform SVD factorization
    - reconstruct the WFA
    """

    def __init__(self, alphabet : List[str], learn_resolution : int):
        """
        Initialize spectral learning with alphabet and prefix/suffix set
        """
        self.basis = KBasis(alphabet, learn_resolution)


    def learn(self, f):
        """
        Perform spectral learning and build WFA
        """
        hp, H, Hs, hs = hankel_blocks_for_function(f, self.basis)

        return Wfa([x for x, _ in self.basis.alphabet()], 2).fit(hp, H, Hs, hs)
