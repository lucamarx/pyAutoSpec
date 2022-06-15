"""
pyAutoSpec

Spectral learning for WFA/MPS
"""

from .wfa import Wfa, SpectralLearning
from .mps import Mps
from .mps2 import Mps2
from .plots import parallel_plot
from .datasets import load_mnist
from .function_wfa import FunctionWfa
from .function_mps import FunctionMps
from .dataset_mps import DatasetMps
from .image_wfa import ImageWfa

__all__ = ["Wfa", "Mps", "Mps2", "parallel_plot", "load_mnist", "SpectralLearning", "FunctionWfa", "FunctionMps", "DatasetMps", "ImageWfa"]

__version__ = "0.8.16"
