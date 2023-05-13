"""
Generate a filled Julia set

Inspired by MIT OCW Course 6.057 Introduction to Matlab
Specifically Homework 3
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


class FilledJulia:
    def __init__(self, z_max: complex, c: complex, iters: int,
                 size: int) -> None:
        """
        :param z_max: Maximum value of the complex number (in both axes)
        :param c: Given value for c
        :param iters: Number of iterations used to see whether the
            iteration with the given c blows up
        :param size: Size of the (square) image
        """
        self.z_max = z_max
        self.c = c
        self.iters = iters
        self.size = size
        self.img = None

    def _generate_matrix(self) -> np.ndarray:
        """
        Generates a self.size x self.size matrix of complex numbers,
        where the real part varies along the x-axis, and the imaginary
        part along the y-axis. The values are in the range
        -z_max ... z_max in both axes.
        :return:
        """
        vals = np.linspace(-self.z_max.real, self.z_max.real, self.size)
        z_arr = np.zeros((self.size, self.size), dtype=complex)
        for j in range(self.size):
            for i in range(self.size):
                z_arr[j, i] = complex(vals[i], vals[j])

        return z_arr

    def _iterate(self, z: complex) -> int:
        """
        Iterates the equation z_n = z_n-1 ** 2 + c
        :param z:
        :return:
        """
        n = 0
        while n < self.iters and abs(z) <= 2:
            z = z * z + self.c
            n += 1
        return n

    @timer
    def _escape_velocity(self, z_arr: np.ndarray) -> np.ndarray:
        """
        Calculate the amount of iterations it takes for the iteration
        to diverge
        :param z_arr: Matrix of complex numbers
        :return: Matrix of the same shape as z_arr, where the escape
            velocity of each item in the matrix is calculated
        """
        vel_arr = np.zeros((self.size, self.size), dtype=np.uint16)
        for j in range(self.size):
            for i in range(self.size):
                vel_arr[j, i] = self._iterate(z_arr[j, i])

        return vel_arr

    def generate_img(self, cmap: str = 'jet') -> None:
        """
        Generates the image
        :param cmap: Name of a colormap supported by matplotlib
        :return:
        """
        z_arr = self._generate_matrix()
        vel_arr = self._escape_velocity(z_arr)
        self.img = np.arctan(0.01 * vel_arr)
        cmap = cm.get_cmap(cmap)
        self.img = cmap(self.img)
        plt.imshow(self.img)
        plt.gca().invert_yaxis()
        plt.show()


def main():
    z_max = 1
    c = -0.297491 + 0.641051j
    iters = 100
    size = 1080
    cmap = 'inferno'
    julia = FilledJulia(z_max, c, iters, size)
    julia.generate_img(cmap)


if __name__ == '__main__':
    main()
