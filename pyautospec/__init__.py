"""
pyAutoSpec

Spectral learning for WFA/MPS/uMPS
"""

from .wfa import Wfa, SpectralLearning
from .mps import Mps
from .mps2 import Mps2
from .umps import UMPS
from .plots import parallel_plot
from .datasets import load_mnist
from .function_wfa import FunctionWfa
from .function_mps import FunctionMps
from .function_umps import FunctionUMps
from .dataset_mps import DatasetMps
from .image_wfa import ImageWfa

__all__ = ["Wfa", "Mps", "Mps2", "UMPS", "parallel_plot", "load_mnist",
           "SpectralLearning", "FunctionWfa", "FunctionMps", "FunctionUMps",
           "DatasetMps", "ImageWfa"]

__version__ = "0.9.1"
