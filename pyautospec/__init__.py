"""
pyAutoSpec

Spectral learning for WFA/MPS
"""

from .wfa import Wfa, SpectralLearning
from .mps import Mps, SymbolicMps
from .function_wfa import FunctionWfa
from .function_mps import FunctionMps
from .image_wfa import ImageWfa

__all__ = ["Wfa", "Mps", "SymbolicMps", "SpectralLearning", "FunctionWfa", "FunctionMps", "ImageWfa"]
