"""
Implement spectral learning algorithm
"""
import numpy as np
import jax.numpy as jnp

from jax import jit
from typing import List
from tqdm.auto import tqdm

from .wfa import Wfa
from .ps_basis import PrefixSuffixBasis, KBasis


@jit
def pseudo_inverse(M):
    """
    To compute the pseudo inverse of M:

    first find its SVD decomposition

    M = U Σ V*

    then the pseudo-inverse is

    M† = V Σ† U*

    see https://en.wikipedia.org/wiki/Moore–Penrose_inverse
    """
    U, S, Vt = jnp.linalg.svd(M, full_matrices=True, compute_uv=True)

    diag = jnp.diag_indices(S.shape[0])
    Sinv = jnp.transpose(jnp.zeros((U.shape[1], Vt.shape[0]), dtype=jnp.float32).at[diag].set(1 / S))

    return jnp.dot(jnp.dot(jnp.transpose(Vt), Sinv), jnp.transpose(U))


def spectral_learning(hp : np.ndarray, H : np.ndarray, Hs : np.ndarray, hs : np.ndarray, n_states : int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform spectral learning of Hankel blocks and build WFA truncating
    SVD expansion to n_states (if specified).

    Parameters:
    -----------

    hp : np.ndarray

    H  : np.ndarray

    Hs : np.ndarray

    hs : np.ndarray

    Returns:
    --------

    The tuple (alpha, A, omega)
    """
    # convert to jax arrays
    hp = jnp.array(hp)
    H  = jnp.array(H)
    Hs = jnp.array(Hs)
    hs = jnp.array(hs)

    # compute full-rank factorization H = P·S
    U, D, V = jnp.linalg.svd(H, full_matrices=True, compute_uv=True)

    # truncate expansion
    rank = jnp.linalg.matrix_rank(H)

    if n_states is not None and n_states < rank:
        rank = n_states

    U, D, V = U[:,0:rank], D[0:rank], V[0:rank,:]
    D_sqrt = jnp.sqrt(D)

    print("Singular values: {}".format(D))

    P = jnp.einsum("ij,j->ij", U, D_sqrt)
    S = jnp.einsum("i,ij->ij", D_sqrt, V)

    # compute pseudo inverses
    Pd, Sd = pseudo_inverse(P), pseudo_inverse(S)

    # α = h·S†
    alpha = jnp.dot(hs, Sd)

    # A = P†·Hσ·S†
    A = jnp.einsum("pi,iaj,jq->paq", Pd, Hs, Sd)

    # ω = P†·h
    omega = jnp.dot(Pd, hp)

    return alpha, A, omega


class SpectralLearning():
    """
    Spectral learning algorithm:

    - generate a base made of words with maximum lenght
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
        p = len(self.basis.prefixes())
        a = len(self.basis.alphabet())
        s = len(self.basis.suffixes())

        # here we use numpy arrays because they are faster to update in place
        hp = np.zeros((p,), dtype=np.float32)
        H  = np.zeros((p,s), dtype=np.float32)
        Hs = np.zeros((p,a,s), dtype=np.float32)
        hs = np.zeros((s,), dtype=np.float32)

        # compute Hankel blocks
        for (u, u_i) in tqdm(self.basis.prefixes()):
            hp[u_i] = f(u)

            for (v, v_i) in self.basis.suffixes():
                hs[v_i] = f(v)

                H[u_i, v_i] = f(u + v)

                for (a, a_i) in self.basis.alphabet():
                    Hs[u_i, a_i, v_i] = f(u + a + v)

        wfa = Wfa([x for x, _ in self.basis.alphabet()], 2)

        wfa.alpha, wfa.A, wfa.omega = spectral_learning(hp, H, Hs, hs)

        return wfa
