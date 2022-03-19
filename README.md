pyAutoSpec
==========

pyAutoSpec is a simple library for demonstrating the **spectral learning
algorithm** detailed in the 2013 paper [Spectral learning of weighted
automata](https://www.cs.upc.edu/~aquattoni/AllMyPapers/mlj_2014.pdf) by Borja
Balle, Xavier Carreras, Franco M. Luque and Ariadna Quattoni.

Installation
------------

To build the library just run

    python setup.py install

Or use the provided docker file:

    docker build -t pyautospec .
    docker run -p 8888:8888 -it pyautospec

and access the Jupyter notebook.

Quick Start
-----------

Suppose you have a function `f` that takes a string and returns a number, you
want to learn a weighted finite automaton that computes `f`, you do this

```python
from pyautospec import SpectralLearning

alphabet = [ord(c) for c in range(ord('a'), ord('z')+1)]

learner = SpectralLearning(alphabet, 3)

automaton = learner.learn(f)
```

see the [Automata Learning](https://github.com/lucamarx/pyAutoSpec/blob/main/examples/Automata Learning.ipynb) example.
