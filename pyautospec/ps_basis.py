"""
Prefix-suffix basis
"""
import abc
import itertools

from typing import Dict, List, Tuple


def factors(x : str, prefix_index : Dict[str, int], suffix_index : Dict[str, int], sigma : str = None) -> List[Tuple[int, int]]:
    if sigma is None:
        if len(x) == 0:
            return [(prefix_index[""], suffix_index[""])]

        pairs = [(prefix_index.get(x[:i]), suffix_index.get(x[i:])) for i in range(len(x)+1)]

        return [(p,s) for p,s in pairs if p is not None and s is not None]

    else:
        if len(x) == 0:
            return []

        if len(x) == 1:
            return (prefix_index[""], suffix_index[""]) if x == sigma else []

        pairs = [(prefix_index.get(x[:i]), suffix_index.get(x[i+1:])) for i in range(len(x)) if x[i] == sigma]

        return [(p,s) for p,s in pairs if p is not None and s is not None]


def complete(alphabet : List[str], prefixes : List[str]) -> List[str]:
    """
    Complete the prefixes: P = P'Σ
    """
    complete_prefixes = set([""])
    for p in set([x[:-1] for x in prefixes if len(x) > 0]):
        for l in alphabet:
            complete_prefixes.add(p+l)

    return list(complete_prefixes)


class PrefixSuffixBasis(metaclass=abc.ABCMeta):
    """
    Abstract base class for all prefix/suffix basis
    """

    @abc.abstractmethod
    def alphabet(self):
        """
        Returns an iterator over alphabet, index pairs: (σ, i)
        """
        pass


    @abc.abstractmethod
    def prefixes(self):
        """
        Returns an iterator over prefix, index pairs: (u, u_i)
        """
        pass


    @abc.abstractmethod
    def prefix_index(self, u : str) -> int:
        """
        Return prefix u's index
        """
        pass


    @abc.abstractmethod
    def suffixes(self):
        """
        Returns an iterator over suffix, index pairs: (v, v_i)
        """
        pass


    @abc.abstractmethod
    def suffix_index(self, v : str) -> int:
        """
        Return suffix v's index
        """
        pass


    def factors(self, x : str, sigma : str = None) -> List[Tuple[int, int]]:
        """
        Decompose x into a prefix/suffix pair

        x = u v

        If sigma is not None decompose x as

        x = u σ v
        """
        pass


class KBasis(PrefixSuffixBasis):
    """
    A basis where the prefix/suffix sets are all the words of length less or
    equal than k
    """

    def __init__(self, alphabet : List[str], k : int):
        """
        Make a basis with P = S = Σ(<k)
        """
        self.k = k
        self._alphabet_index = {alphabet[i]: i for i in range(0, len(alphabet))}

        ps_set = [""]
        for l in range(1, k+1):
            ps_set.extend([''.join(w) for w in itertools.product(*([alphabet] * l))])

        self._ps_index = {ps_set[i]: i for i in range(0, len(ps_set))}


    def __repr__(self) -> str:
        return "KBasis({})".format(self.k)


    def alphabet(self):
        return self._alphabet_index.items()


    def prefixes(self):
        return self._ps_index.items()


    def prefix_index(self, u : str) -> int:
        return self._ps_index.get(u)


    def suffixes(self):
        return self._ps_index.items()


    def suffix_index(self, v : str) -> int:
        return self._ps_index.get(v)


    def factors(self, x : str, sigma = None) -> List[Tuple[int, int]]:
        return factors(x, self._ps_index, self._ps_index, sigma)


class ClosureBasis(PrefixSuffixBasis):
    """
    The closure basis of a set of words.
    """

    def __init__(self, alphabet : List[str], words : List[str]):
        """
        """
        self._alphabet_index = {alphabet[i]: i for i in range(0, len(alphabet))}

        prefixes, suffixes = set(), set()
        for x in words:
            for i in range(0,len(x)+1):
                prefixes.add(x[:i])
                suffixes.add(x[i:])

        prefixes = sorted(complete(alphabet, list(prefixes)))
        suffixes = sorted(suffixes)

        self._prefix_index = {prefixes[i]: i for i in range(0, len(prefixes))}
        self._suffix_index = {suffixes[i]: i for i in range(0, len(suffixes))}


    def __repr__(self) -> str:
        return "ClosureBasis() prefixes:{}, suffixes:{}".format(len(self._prefix_index), len(self._suffix_index))


    def alphabet(self):
        return self._alphabet_index.items()


    def prefixes(self):
        return self._prefix_index.items()


    def prefix_index(self, u : str) -> int:
        return self._prefix_index.get(u)


    def suffixes(self):
        return self._suffix_index.items()


    def suffix_index(self, v : str) -> int:
        return self._suffix_index.get(v)


    def factors(self, x : str, sigma = None) -> List[Tuple[int, int]]:
        return factors(x, self._prefix_index, self._suffix_index, sigma)
