"""
Multi-dimensional vector-word encoder
"""
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


    def __repr__(self):
        return f"{self.dim}-dimensional encoder"


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
