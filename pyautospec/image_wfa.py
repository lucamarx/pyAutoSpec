"""
Wfa based image compression algorithm
"""
import itertools
import numpy as np

from tqdm.auto import tqdm
from skimage import color
from .spectral_learning import SpectralLearning


def decode(w : str, tl=(0, 0), sz=100):
    """
    Decode word → x-y coordinates, square size
    """
    if w == "":
        return tl, sz

    sz = sz // 2

    if w[0] == "b":
        tl = (tl[0]+sz, tl[1])
    elif w[0] == "c":
        tl = (tl[0], tl[1]+sz)
    elif w[0] == "d":
        tl = (tl[0]+sz, tl[1]+sz)

    return decode(w[1:], tl, sz)


def get_image(image, w : str):
    (x, y), s = decode(w, sz=image.shape[0])

    # average image over square
    return image[x, y] if s == 0 else np.average(image[x:(x+s), y:(y+s)])


class ImageWfa():

    def __init__(self, image : np.ndarray, learn_resolution : int = 3):
        """
        Intialize image model
        """
        # convert to grayscale
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image = color.rgb2gray(image)
            elif image.shape[2] == 4:
                image = color.rgb2gray(color.rgba2rgb(image))
            else:
                raise Exception("Invalid image shape")

        # crop
        w, h = image.shape
        if w > h:
            image = image[(w-h) // 2 : h + (w-h) // 2, :]
        elif w < h:
            image = image[:, (h-w) // 2 : w + (h-w) // 2]

        self.image = image / np.max(image)
        self.splrn = SpectralLearning(["a", "b", "c", "d"], learn_resolution)
        self.model = self.splrn.learn(lambda w: get_image(self.image, w))


    def __repr__(self):
        return "WFA(states={}) {}x{}".format(len(self.model), self.image.shape[0], self.image.shape[1])


    def reconstruct(self, resolution : int = 5):
        """
        Reconstruct the image
        """
        image = np.zeros(self.image.shape)
        error, count = 0, 0
        for w in tqdm([''.join(w) for w in itertools.product(*([["a", "b", "c", "d"]] * resolution))]):
            (x, y), s = decode(w, sz=image.shape[0])

            v0 = self.image[x, y] if s == 0 else np.average(self.image[x:(x+s), y:(y+s)])
            v1 = self.model.evaluate(w)

            error += abs(v1 - v0)
            count += 1

            image[x:(x+s), y:(y+s)] = v1

        image[image < 0] = 0

        return image, error/count
