"""
Various wfa/mps plots
"""
import numpy as np
import jax.numpy as jnp
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from matplotlib.patches import PathPatch
from matplotlib.path import Path
from functools import reduce
from typing import List


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def mps_tensor_hist(mps, bins : int = 20):
    """
    Plot the distribution of tensor entries
    """
    plt.figure()
    plt.title("MPS Tensor Entries Distribution")

    entries = reduce(lambda x,y: np.concatenate((x,y)), [A.flatten() for A in mps])

    plt.hist(entries, bins=bins)


def transition_plot(n_states : int, word_len : int, paths : np.ndarray, weights : np.ndarray, title : str = ""):
    """
    Plot weight contributions
    """
    _, host = plt.subplots(figsize=(12,4))

    host.set_title(title, fontsize=18, pad=25)
    host.set_xlim(0, word_len)
    host.set_xticks([], labels=None)

    axes = [host] + [host.twinx() for _ in range(word_len)]

    for i, ax in enumerate(axes):
        ax.set_ylim(0, n_states-1)
        ax.set_yticks(range(n_states), labels=None)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        if ax != host:
            ax.spines['left'].set_visible(False)
            ax.spines["right"].set_position(("axes", i / word_len))

    colors = plt.cm.Set2.colors
    max_weight = np.max(np.abs(weights))

    pws = []
    for i in range(paths.shape[0]):
        pws.append(([(j, paths[i,j].item()) for j in range(paths.shape[1])], weights[i].item()))

    c = 0
    for path in pws:
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


def function_mps_comparison_chart(mps, n_points : int = 50, paths_threshold : float = None):
    """
    Compare the learned wfa with the original function
    """
    xs = jnp.linspace(mps.x0, mps.x1, endpoint = False, num = n_points)

    v0 = jnp.array([mps.f(x) for x in xs])
    v1 = jnp.array([mps(x) for x in xs])

    error = jnp.abs(v1 - v0)

    _, ax1 = plt.subplots()

    plt.title("{} reconstruction error: avg={:.2f} max={:.2f} ".format(mps.f.__repr__(), jnp.average(error), jnp.max(error)))

    ax1.set_xlabel("x")
    ax1.set_ylabel("f(x)")

    p1 = ax1.plot(xs, v0, color="green", label="original f")
    p2 = ax1.plot(xs, v1, color="orange", label="f")
    ls = p1 + p2

    if paths_threshold is not None:
        ax2 = ax1.twinx()
        ax2.set_ylabel("contrib. paths")
        ls += ax2.plot(xs, [mps.paths_weights(x, threshold=paths_threshold)[0].shape[0] for x in xs], color="blue", label="paths n.")

    plt.legend(ls, [l.get_label() for l in ls], loc=0)


def function_mps_path_value_chart(mps):
    """
    Plot contributions to the final value by paths and function argument
    """
    all_paths, all_xencs = mps._all_paths(), mps._all_encodings()

    W = np.zeros((len(all_paths), len(all_xencs)))

    i = 0
    for path in all_paths:
        j = 0
        for x in all_xencs:
            W[i,j] = mps.path_state_weight(path, x[0])
            j += 1
        i += 1

    W = np.abs(W)
    W = W / np.max(W)

    plt.figure()
    plt.title("Path/Value Contributions")

    plt.imshow(W, cmap=cm.magma, origin='lower', aspect="{:2f}".format(len(all_xencs) / len(all_paths)))


def mps_entanglement_entropy_chart(mps):
    """
    Plot entanglement entropy chart
    """
    plt.figure()

    plt.title('Left/Right Entanglement Entropy')

    plt.xlabel('left-right block sizes')
    plt.ylabel('entropy')

    S = mps.entanglement_entropy()

    plt.bar(range(1, len(S)+1), S,
        tick_label=["{}-{}".format(l, len(S)+1-l) for l in range(1, len(S)+1)])


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
