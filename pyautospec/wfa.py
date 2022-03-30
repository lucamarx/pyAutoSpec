"""
Weighted finite automaton implemented as a matrix product state
"""
import graphviz
import itertools
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from typing import List
from jax import jit, vmap


@jit
def evaluate(alpha, A, omega, X):
    y = alpha
    for i in range(0, len(X)):
        y = jnp.dot(y, A[:,X[i],:])
    return jnp.dot(y, omega)


@jit
def path_weight(alpha, A, omega, X, path):
    """
    Compute the weight of a path in the state graph corresponding to a word X
    """
    weight = alpha[path[0]]

    for i in range(1, path.shape[0]):
        p, q = path[i-1], path[i]
        weight *= A[p, X[i-1], q]

    weight *= omega[path[path.shape[0] - 1]]

    return weight


def word_paths(alpha, A, omega, X, threshold):
    """
    Enumerate all paths contributing to final weight
    """
    paths = jnp.array(list(itertools.product(*([range(0, alpha.shape[0])] * (1 + len(X))))), dtype=jnp.int32)
    weights = vmap(lambda p: path_weight(alpha, A, omega, X, p), in_axes=0)(paths)

    indexes = jnp.where(weights > threshold)[0]
    paths, weights = paths[indexes,:], weights[indexes]

    res = []
    for i in range(0, indexes.shape[0]):
        res.append(([(j, paths[i,j].item()) for j in range(0, paths.shape[1])], weights[i].item()))

    return res


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


    def __len__(self):
        """
        The length is just the number of states
        """
        return self.A.shape[0]


    def set_starting_state(self, p : int, w : float = 1):
        """
        Set state p as starting with weight w
        """
        self.alpha = self.alpha.at[p].set(w)


    def add_transition(self, p : int , l : str, w : float, q : int):
        """
        Add transition p -l-> q with weight w
        """
        self.A = self.A.at[p, self.alphabet_map[l], q].set(w)


    def set_accepting_state(self, p : int, w : float = 1.0):
        """
        Set state p as accepting with weight w
        """
        self.omega = self.omega.at[p].set(w)


    def __call__(self, X):
        """
        Evaluate WFA over word X
        """
        return evaluate(self.alpha, self.A, self.omega, [self.alphabet_map[a] for a in X]).item()


    def evaluate(self, X):
        """
        Evaluate WFA over word X
        """
        return evaluate(self.alpha, self.A, self.omega, [self.alphabet_map[a] for a in X]).item()


    def diagram(self, title="WFA"):
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


    def transition_plot(self, X, threshold=1e-4):
        """
        Plot weight contributions
        """
        fig, host = plt.subplots(figsize=(12,4))

        host.set_title("'{}' → {:.4f}".format(X, self.evaluate(X)), fontsize=18, pad=25)
        host.set_xlim(0, len(X))
        host.set_xticks([], labels=None)

        axes = [host] + [host.twinx() for _ in range(len(X))]

        for i, ax in enumerate(axes):
            ax.set_ylim(0, len(self) - 1)
            ax.set_yticks(range(len(self)), labels=None)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if ax != host:
                ax.spines['left'].set_visible(False)
                ax.spines["right"].set_position(("axes", i / len(X)))

        paths = word_paths(self.alpha, self.A, self.omega, [self.alphabet_map[a] for a in X], threshold)
        colors = plt.cm.Set2.colors
        max_weight = max([abs(p[1]) for p in paths])

        c = 0
        for path in paths:
            verts = path[0]
            codes = [Path.MOVETO] + [Path.LINETO for _ in range(len(verts) - 1)]
            weight = path[1]

            host.add_patch(PathPatch(Path(verts, codes),
                                     lw=int(90 * abs(weight) / max_weight),
                                     alpha=0.2,
                                     facecolor='none',
                                     edgecolor=colors[c % len(colors)]))
            c += 1

        plt.tight_layout()
        plt.show()
