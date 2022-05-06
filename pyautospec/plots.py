"""
Various wfa plots
"""
import itertools
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from typing import List
from jax import jit, vmap

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


def transition_plot(wfa, X : List[str], threshold : float = 1e-4):
    """
    Plot weight contributions
    """
    _, host = plt.subplots(figsize=(12,4))

    host.set_title("'{}' → {:.4f}".format(X, wfa(X)), fontsize=18, pad=25)
    host.set_xlim(0, len(X))
    host.set_xticks([], labels=None)

    axes = [host] + [host.twinx() for _ in range(len(X))]

    for i, ax in enumerate(axes):
        ax.set_ylim(0, len(wfa) - 1)
        ax.set_yticks(range(len(wfa)), labels=None)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.spines["right"].set_position(("axes", i / len(X)))

    paths = word_paths(wfa.alpha, wfa.A, wfa.omega, [wfa.alphabet_map[a] for a in X], threshold)
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


def function_wfa_comparison_chart(wfa, n_points : int = 50, resolution : int = 12, plot_derivative : bool = False):
    """
    Compare the learned wfa with the original function
    """
    xs = jnp.linspace(wfa.x0, wfa.x1, endpoint = False, num = n_points)

    v0 = jnp.array([wfa.f(x) for x in xs])
    v1 = jnp.array([wfa(x, resolution) for x in xs])

    error = jnp.abs(v1 - v0)

    plt.figure()

    plt.title("{} reconstruction error: avg={:.2f} max={:.2f} ".format(wfa.f.__repr__(), jnp.average(error), jnp.max(error)))

    plt.plot(xs, v0, label="original f")
    plt.plot(xs, v1, label="f")

    if plot_derivative:
        v2 = jnp.array([wfa.prime(x, resolution) for x in xs])
        plt.plot(xs, v2, label="f'")

    plt.legend()
