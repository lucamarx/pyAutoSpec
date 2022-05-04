"""
pyAutoSpec

Spectral learning for WFA/MPS
"""

from .mps import Mps
from .wfa import Wfa
from .spectral_learning import SpectralLearning
from .function_wfa import FunctionWfa
from .image_wfa import ImageWfa

__all__ = ["Mps", "Wfa", "SpectralLearning", "FunctionWfa", "ImageWfa"]
