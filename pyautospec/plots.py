"""
Various wfa plots
"""
import itertools
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from typing import List
from jax import jit, vmap


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


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
    if resolution is None:
        v1 = jnp.array([wfa(x) for x in xs])
    else:
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


def parallel_plot(X : np.ndarray, y : np.ndarray, feature_names : List[str] = None, target_names : List[str] = None, title : str = "Parallel Plot"):
    """
    Plot multidimensional dataset as a parallel plot
    """
    if feature_names is None:
        feature_names = ["f {}".format(i+1) for i in range(X.shape[1])]

    if target_names is None:
        target_names = ["t {}".format(i+1) for i in range(y.max()+1)]

    Xmin, Xmax = X.min(axis=0), X.max(axis=0)
    Xrange = Xmax - Xmin

    # transform all data to be compatible with the main axis
    Z = np.zeros_like(X)
    Z[:, 0]  = X[:, 0]
    Z[:, 1:] = (X[:, 1:] - Xmin[1:]) / Xrange[1:] * Xrange[0] + Xmin[0]

    _, host = plt.subplots(figsize=(10,4))

    axes = [host] + [host.twinx() for _ in range(X.shape[1] - 1)]
    for i, ax in enumerate(axes):
        ax.set_ylim(Xmin[i], Xmax[i])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        if ax != host:
            ax.spines["left"].set_visible(False)
            ax.spines["right"].set_position(("axes", i / (X.shape[1] - 1)))
            ax.yaxis.set_ticks_position("right")

    host.set_xlim(0, X.shape[1] - 1)

    host.set_xticks(range(X.shape[1]))
    host.set_xticklabels(feature_names, fontsize=14)
    host.tick_params(axis='x', which="major", pad=7)
    host.xaxis.tick_top()
    host.spines["right"].set_visible(False)

    host.set_title(title, fontsize=18, pad=12)

    colors = plt.cm.Set2.colors
    legend_handles = [None for _ in target_names]
    for j in range(X.shape[0]):
        # create bezier curves
        verts = list(zip([x for x in np.linspace(0, len(X) - 1, len(X) * 3 - 2, endpoint=True)], np.repeat(Z[j, :], 3)[1:-1]))
        codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
        patch = PathPatch(Path(verts, codes), facecolor="none", lw=1, alpha=0.6, edgecolor=colors[y[j]])

        host.add_patch(patch)
        legend_handles[y[j]] = patch

    host.legend(legend_handles, target_names, loc="lower center",
                bbox_to_anchor=(0.5, -0.18), ncol=len(target_names), fancybox=True,
                shadow=True)

    plt.tight_layout()
    plt.show()


def training_chart(train_costs : List[int], valid_costs : List[int]):
    """
    Chart the training/validation performance
    """
    plt.figure()

    plt.title("Training/Validation Losses")

    plt.plot(moving_average(np.array(train_costs), 10), label="training")

    if len(valid_costs) > 0:
        plt.plot(moving_average(np.array(valid_costs), 10), label="validation")

    plt.legend()
