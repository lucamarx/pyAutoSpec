"""
Implement spectral learning algorithm
"""
import itertools
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

    def __init__(self, alphabet : List[str], prefix_suffix_length : int):
        """
        Initialize spectral learning with alphabet and prefix/suffix set
        """
        self.alphabet = alphabet
        self.alphabet_index = {alphabet[i]: i for i in range(0, len(alphabet))}

        prefix_suffix_set = []
        for l in range(1, prefix_suffix_length+1):
            prefix_suffix_set.extend([''.join(w) for w in itertools.product(*([alphabet] * l))])

        self.prefix_index = {prefix_suffix_set[i]: i for i in range(0, len(prefix_suffix_set))}


    def learn(self, f):
        """
        Perform spectral learning and build WFA
        """
        d = len(self.prefix_index)
        a = len(self.alphabet_index)

        h  = jnp.zeros((d,), dtype=jnp.float32)
        H  = jnp.zeros((d,d), dtype=jnp.float32)
        Hs = jnp.zeros((d,a,d), dtype=jnp.float32)

        # compute Hankel blocks
        for (u, u_i) in tqdm(self.prefix_index.items()):
            h = h.at[u_i].set(f(u))

            for (v, v_i) in self.prefix_index.items():
                H = H.at[u_i, v_i].set(f(u + v))

                for (a, a_i) in self.alphabet_index.items():
                    Hs = Hs.at[u_i, a_i, v_i].set(f(u + a + v))

        # compute full-rank factorization H = P·S
        u, s, v = jnp.linalg.svd(H, full_matrices=True, compute_uv=True)

        # truncate expansion
        # rank = jnp.argmin(jnp.abs(s) > 1e-5).item()
        rank = jnp.linalg.matrix_rank(H)

        P, S = jnp.einsum("ij,j->ij", u[:,0:rank], s[0:rank]), v[0:rank,:]

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
