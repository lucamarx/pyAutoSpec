pyAutoSpec
==========

pyAutoSpec is a simple library for demonstrating the **spectral learning
algorithm** detailed in the 2014 paper [Spectral learning of weighted
automata](https://www.cs.upc.edu/~aquattoni/AllMyPapers/mlj_2014.pdf) by Borja
Balle, Xavier Carreras, Franco M. Luque and Ariadna Quattoni.

You can find an introduction on my
[blog](https://lucamarx.com/blog/2022/0323-spectral_learning/).

Disclaimer
----------

This is a **toy** implementation of spectral learning:

- it is not optimized
- it is not thoroughly tested
- it is as simple as possible

**DO NOT USE IN PRODUCTION**

If you need spectral learning in your application use
[scikit-splearn](https://pypi.org/project/scikit-splearn/) instead.

Installation
------------

To build the library just run

    pip install -e .

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

see the [Automata Learning](<https://github.com/lucamarx/pyAutoSpec/blob/main/examples/Automata Learning.ipynb>) example.


The same thing can be done for a function `f` that takes a number in an interval
and returns a number, in this case you can do this

```python
from math import sin, pi
from pyautospec import FunctionWfa

# learn the sin function in the [0,2π] interval
sin_aut = FunctionWfa().fit(sin, x0=0.0, x1=2*pi, learn_resolution=2)
```

see the [Trigonometric Functions](<https://github.com/lucamarx/pyAutoSpec/blob/main/examples/Trigonometric Functions.ipynb>) example.
