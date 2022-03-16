"""
pyAutoSpec

Spectral learning for WFA
"""

from .wfa import Wfa
from .spectral_learning import SpectralLearning
from .image_wfa import ImageWfa
from .function_wfa import FunctionWfa


__all__ = ["Wfa", "SpectralLearning", "ImageWfa", "FunctionWfa"]
