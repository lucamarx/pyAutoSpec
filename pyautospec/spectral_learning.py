"""
Implement spectral learning algorithm
"""
import itertools
import numpy as np
import jax.numpy as jnp
from typing import List
from tqdm.auto import tqdm
from jax import jit

from .wfa import Wfa


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


class SpectralLearning():
    """
    Spectral learning algorithm:

    - generate a base made of words with maximum lenght
    - build the Hankel matrix for the function f
    - perform SVD factorization
    - reconstruct the WFA
    """

    def __init__(self, alphabet : List[str], prefix_suffix_length : int):
        """
        Initialize spectral learning with alphabet and prefix/suffix set
        """
        self.alphabet = alphabet
        self.alphabet_index = {alphabet[i]: i for i in range(0, len(alphabet))}

        # the prefix/suffix set is made of words with a maximum length
        prefix_suffix_set = [""]
        for l in range(1, prefix_suffix_length+1):
            prefix_suffix_set.extend([''.join(w) for w in itertools.product(*([alphabet] * l))])

        self.prefix_index = {prefix_suffix_set[i]: i for i in range(0, len(prefix_suffix_set))}


    def learn(self, f):
        """
        Perform spectral learning and build WFA
        """
        d = len(self.prefix_index)
        a = len(self.alphabet_index)

        # here we use numpy arrays because they are faster to update in place
        h  = np.zeros((d,), dtype=np.float32)
        H  = np.zeros((d,d), dtype=np.float32)
        Hs = np.zeros((d,a,d), dtype=np.float32)

        # compute Hankel blocks
        for (u, u_i) in tqdm(self.prefix_index.items()):
            h[u_i] = f(u)

            for (v, v_i) in self.prefix_index.items():
                H[u_i, v_i] = f(u + v)

                for (a, a_i) in self.alphabet_index.items():
                    Hs[u_i, a_i, v_i] = f(u + a + v)

        # convert to jax arrays
        h  = jnp.array(h)
        H  = jnp.array(H)
        Hs = jnp.array(Hs)

        # compute full-rank factorization H = P·S
        U, D, V = jnp.linalg.svd(H, full_matrices=True, compute_uv=True)

        # truncate expansion
        rank = jnp.linalg.matrix_rank(H)
        U, D, V = U[:,0:rank], D[0:rank], V[0:rank,:]
        D_sqrt = jnp.sqrt(D)

        P = jnp.einsum("ij,j->ij", U, D_sqrt)
        S = jnp.einsum("i,ij->ij", D_sqrt, V)

        # compute pseudo inverses
        Pd, Sd = pseudo_inverse(P), pseudo_inverse(S)

        # compute WFA parameters
        wfa = Wfa(self.alphabet, rank)

        # α = h·S†
        wfa.alpha = jnp.dot(h, Sd)

        # A = P†·Hσ·S†
        wfa.A = jnp.einsum("pi,iaj,jq->paq", Pd, Hs, Sd)

        # ω = P†·h
        wfa.omega = jnp.dot(Pd, h)

        return wfa
