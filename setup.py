"""
pyAutoSpec module setup
"""
from setuptools import setup

VERSION = "0.3.1"

setup(
    name="pyAutoSpec",

    license="MIT",
    version=VERSION,

    description="Simple spectral learning for weighted automata",
    long_description="""

    A python library to demonstrate the spectral learning algorithm for weighted
    finite automata.

    """,
    # long_description_content_type="text/x-rst",

    author="Luca Marx",
    author_email="luca@lucamarx.com",

    url="https://github.com/lucamarx/pyAutoSpec",

    packages=["pyautospec"],

    python_requires=">=3.8",

    install_requires=[
        "graphviz>=0.19.0",
        "tqdm>=4.62.0",
        "jax>=0.3.0",
    ],
    # extras_require={
    #     "test": ["unittest"]
    # }
)
