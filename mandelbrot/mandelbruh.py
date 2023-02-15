"""
Some sort of illustration of the Mandelbrot set
"""

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from typing import Callable
from timeit import default_timer


def timer(func: Callable):
    """
    Decorator to time a function
    :param func: Function to be timed
    :return:
    """
    def wrapper(*args, **kwargs):
        s = default_timer()
        rv = func(*args, **kwargs)
        print(f'Time: {func.__name__}: {default_timer() - s:.4f} s')
        return rv
    return wrapper


class Mandelbruh:
    def __init__(self, width: int = 400, height: int = 200,
                 iter_limit: int = 100) -> None:
        """
        Initialize the set
        :param width: Width of the image
        :param height: Height of the image
        :param iter_limit: Max. number of iterations that are checked
        before a complex number is determined to be within the
        Mandelbrot set (must be less than 2^16, 100 should be enough)
        """
        self.w = width
        self.h = height
        self.iter_limit = iter_limit
        if self.iter_limit < 255:
            self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        else:
            self.img = np.zeros((self.h, self.w, 3), dtype=np.uint16)

    def _scale_pixel(self, ind: int, axis: int) -> float:
        """
        Scales the pixel index to fit within the Mandelbrot set's limits,
        -2.5 < x < 1, -1 < y < 1
        :param ind: Pixel index in either x- or y-direction
        :param axis: The axis of the pixel, axis == 0 indicates x-axis,
        axis == 1 indicates y-axis
        :return:
        """
        if axis == 0:
            mini, maxi = -2.5, 1
            max_range = maxi - mini
            ind /= self.w
            return mini + max_range * ind
        else:
            mini, maxi = -1.12, 1.12
            max_range = maxi - mini
            ind /= self.h
            return mini + max_range * ind

    def _iterate(self, px: float, py: float) -> int:
        """
        Iterates over the Mandelbrot function without using
        actual complex numbers
        :param px: Scaled index of a pixel in the x-direction
        :param py: Scaled index of a pixel in the y-direction
        :return: The number of iterations it took for (possible)
        divergence
        """
        count = 0
        x, y = 0.0, 0.0
        while x * x + y * y <= 2 * 2 and count < self.iter_limit:
            xtemp = x * x - y * y + px
            y = 2 * x * y + py
            x = xtemp
            count += 1
        return count

    @timer
    def _find_valid_pixels(self) -> None:
        """
        Finds the pixels that belong to the Mandelbrot set
        :return:
        """
        for y in range(self.img.shape[0]):
            y_s = self._scale_pixel(y, axis=1)
            for x in range(self.img.shape[1]):
                x_s = self._scale_pixel(x, axis=0)
                self.img[y, x, 2] = self._iterate(x_s, y_s)

    def generate_img(self, col_map: str,
                     set_color: tuple = None) -> None:
        """
        Generates the image according to the Mandelbrot set and shows
        the generated image
        :param col_map: The name of a colormap supported by matplotlib
        :param set_color: If provided, the pixels within the set
        will be colored by the given color
        :return:
        """
        self._find_valid_pixels()
        cmap = cm.get_cmap(col_map)
        if set_color is not None:
            set_inds = np.where(self.img[:, :, 2] == self.iter_limit)
            other_inds = np.where(self.img[:, :, 2] != self.iter_limit)
            self.img[set_inds] = set_color
            others = np.arctan(0.015 * self.img[other_inds[0], other_inds[1], 2])
            self.img[other_inds] = cmap(others)[:, 0:3] * 255
        else:
            self.img = cmap(np.arctan(0.015 * self.img[:, :, 2]))
        plt.imshow(self.img)
        plt.xticks([])
        plt.yticks([])
        plt.show()


def main():
    w, h = 2560, 1440
    iter_limit = 250
    cmap = 'jet'
    black = (0, 0, 0)
    m = Mandelbruh(w, h, iter_limit)
    m.generate_img(cmap, set_color=black)


if __name__ == '__main__':
    main()
