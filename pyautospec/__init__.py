"""
pyAutoSpec

Spectral learning for WFA/MPS/uMPS
"""

from .wfa import Wfa, SpectralLearning
from .mps import Mps
from .mps2 import Mps2
from .umps import UMPS
from .umpo import UMPO
from .plots import parallel_plot
from .datasets import load_mnist
from .function_wfa import FunctionWfa
from .function_mps import FunctionMps
from .function_umps import FunctionUMps
from .function_umpo import FunctionUMpo
from .gen_fun_umps import GenFunctionUMps
from .dataset_mps import DatasetMps
from .dataset_umps import DatasetUMps
from .image_wfa import ImageWfa

__all__ = ["Wfa", "Mps", "Mps2", "UMPS", "UMPO", "parallel_plot", "load_mnist",
           "SpectralLearning", "FunctionWfa", "FunctionMps", "FunctionUMps",
           "FunctionUMpo", "GenFunctionUMps", "DatasetMps", "DatasetUMps",
           "ImageWfa"]

__version__ = "0.9.5"
