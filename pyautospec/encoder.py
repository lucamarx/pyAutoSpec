"""
Multi-dimensional vector-word encoder
"""
from __future__ import annotations

import numpy as np

from math import floor, log2
from typing import List, Tuple, Optional


def _scalar2word(x : float, l : int, limits : Tuple[float, float]) -> List[int]:
    x0, x1 = limits

    if x < x0 or x >= x1:
        raise Exception("outside limits")

    x = (x - x0) / (x1 - x0)
    w = []
    for _ in range(l+1):
        d = 1 if x >= 1 else 0
        w += [d]
        x = (x - d)*2
    return w[1:]


def _word2scalar(w : List[int], limits : Tuple[float, float]) -> float:
    x0, x1 = limits
    w = [0] + w
    return x0 + sum([int(w[i]) * 2**(-i) for i in range(len(w))]) * (x1-x0)


def _binary2digit(b : Tuple) -> int:
    return sum(map(lambda x: x[1] * 2**x[0], enumerate(b)))


def _digit2binary(d : int, dim : int) -> Tuple:
    return tuple([1 if (d & 2**i) != 0 else 0 for i in range(dim)])


class VectorEncoder():
    """Multi-dimensional vector-word encoder"""

    def __init__(self, limits : List[Tuple[float, float]], encoding_length : Optional[int] = 8):
        """Create a vector encoder

        `(x, y, ...) → w`

        the word here is intended to be an l-word, that is a list of particle
        dimensions

        Parameters
        ----------

        limits : List[Tuple[float,float]]
        The limits of each vector dimension

        encoding_length : int, optional
        The length of the encoded word

        """
        self.dim = len(limits)
        self.limits = limits
        self.part_d = 2**self.dim
        self.encoding_length = encoding_length


    def __eq__(self, other : VectorEncoder) -> bool:
        if self.dim != other.dim:
            return False

        if self.encoding_length != other.encoding_length:
            return False

        return all([l1 == l2 for l1,l2 in zip(self.limits, other.limits)])


    def __repr__(self):
        return f"{self.dim}-dimensional encoder"


    def one_hot(self, X : List[List[int]]) -> np.ndarray:
        """Perform one-hot encoding

        Parameters
        ----------

        X : np.ndarray

        """
        idxs = np.array(X).reshape(-1)
        return np.eye(self.part_d)[idxs].reshape((-1, self.encoding_length, self.part_d))


    def encode(self, *args) -> List[int]:
        """Encode a vector into an l-word

        Parameters
        ----------

        *args : float
        The vector components

        length : int
        The length of the encoded l-word

        Returns
        -------

        The encoded l-word

        """
        return list(map(_binary2digit, zip(*[_scalar2word(args[d], l=self.encoding_length, limits=self.limits[d]) for d in range(self.dim)])))


    def decode(self, word : List[int]) -> Tuple:
        """Decode a l-word into a vector

        Parameters
        ----------

        word : List[int]
        The l-word to be decoded

        Returns
        -------

        `(x,y,...)`

        A tuple of the vector components

        """
        if len(word) == 0:
            return tuple([x0 for (x0,_) in self.limits])

        comps = list(map(list, zip(*[_digit2binary(d, dim=self.dim) for d in word])))
        return tuple([_word2scalar(comps[d], self.limits[d]) for d in range(self.dim)])


    def resolution(self) -> float:
        """The encoding resolution
        """
        return [2**(-self.encoding_length) * (x1-x0) for x0,x1 in self.limits]


    def encode_array(self, X : np.ndarray) -> np.ndarray:
        """Encode a vector or a batch of vectors into a v-word or a batch of
        v-words

        Parameters
        ----------

        X : np.ndarray
        A vector or a batch of vectors

        Returns
        -------

        A v-word or a batch of v-words

        """
        if len(X.shape) not in [1,2]:
            raise Exception("invalid X shape: must be a vector or a batch of vectors")

        if len(X.shape) == 1:
            X = X.reshape((1,-1))

        if X.shape[1] != len(self.limits):
            raise Exception(f"invalid shape: vectors must be {len(self.limits)}-dimensional")

        return np.array([self.encode(*x) for x in X])


    def decode_array(self, W : np.ndarray) -> np.ndarray:
        """Decode a v-word or a batch of v-words into a vactor or a batch of
        vectors

        Parameters
        ----------

        W : np.ndarray
        A v-word or a batch ov v-words

        Returns
        -------

        A vector or a batch of vectors

        """
        if len(W.shape) not in [1,2]:
            raise Exception("invalid W shape: must be a v-vector or a batch of v-vectors")

        if len(W.shape) == 1:
            W = W.reshape((1,-1))

        if W.shape[1] != self.encoding_length:
            raise Exception(f"invalid shape: v-vectors must be {self.encoding_length} long")

        return np.array([self.decode(w) for w in W])


class IntegerEncoder:
    """Integer encoder"""


    def __repr__(self):
        return "int encoder"


    def encode(self, n : int, length : Optional[int] = None) -> List[int]:
        """Encode an integer into an l-word

        Parameters
        ----------

        n : int
        The integer

        length : int, optional
        The length of the encoded l-word

        Returns
        -------

        The encoded l-word

        """
        if length is None:
            length = int(floor(log2(n)))+1 if n>0 else 1

        w = []
        for _ in range(length):
            w += [n % 2]
            n = n // 2
        return w


    def decode(self, word : List[int]) -> int:
        """Decode a l-word into an integer

        Parameters
        ----------

        word : List[int]
        The l-word to be decoded

        Returns
        -------

        The decoded integer

        """
        return sum([word[i] * 2**i for i in range(len(word))])
