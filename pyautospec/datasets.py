"""
Utilities to load datasets
"""
import numpy as np
import os, gzip, hashlib

from tqdm.auto import tqdm
from urllib.request import urlopen


def fetch(url, temp_dir):
    fp = os.path.join(temp_dir, hashlib.md5(url.encode('utf-8')).hexdigest())

    if os.path.isfile(fp):
        with open(fp, "rb") as f:
            data = f.read()

    else:
        with urlopen(url) as f:
            data = f.read()

        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)

        with open(fp, "wb") as f:
            f.write(data)

    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def load_mnist(N=None, fmt=None, test=False, temp_dir="./.pyautospec_data"):
    """
    Load the MNIST dataset

    Parameters:
    -----------
    N : int
    if it is not None sample N items

    fmt : str


    Returns:
    --------
    X, y
    """
    t = "t10k" if test else "train"
    X = fetch("http://yann.lecun.com/exdb/mnist/{}-images-idx3-ubyte.gz".format(t), temp_dir)[0x10:].reshape((-1, 28, 28))
    y = fetch("http://yann.lecun.com/exdb/mnist/{}-labels-idx1-ubyte.gz".format(t), temp_dir)[8:]

    if N is not None:
        B = np.random.randint(0, X.shape[0], size=N)
        X, y = X[B,:,:], y[B]

    if fmt is None:
        X1 = X

    elif fmt == "16x16":
        X1 = np.zeros((X.shape[0], 16, 16))
        for b in tqdm(range(X.shape[0])):
            for i in range(14):
                for j in range(14):
                    X1[b,i+1,j+1] = np.average(X[b, 2*i:2*i+2, 2*j:2*j+2])

    elif fmt == "14x14":
        X1 = np.zeros((X.shape[0], 14, 14))
        for b in tqdm(range(X.shape[0])):
            for i in range(14):
                for j in range(14):
                    X1[b,i,j] = np.average(X[b, 2*i:2*i+2, 2*j:2*j+2])

    else:
        raise Exception("invalid format")

    return X1, y
