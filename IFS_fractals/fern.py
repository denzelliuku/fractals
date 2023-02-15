"""
Plots the Barnsley Fern

Inspired by Coding Math's youtube video:
https://www.youtube.com/watch?v=geqq63WFLr0&t
"""

import numpy as np
import matplotlib.pyplot as plt

from random import random
from fern_rules import rules


def get_rule() -> dict:
    """
    Returns a random rule based on their weight
    :return:
    """
    rand = random()
    for rule in rules:
        if rand < rule['weight']:
            return rule
        rand -= rule['weight']


def create_points(n: int = 100) -> np.ndarray:
    """
    Creates the points making up the fractal
    :param n: Number of points
    :return: Coordinates of the points
    """
    points = np.ones((n, 3))
    x, y = random(), random()
    points[0, :2] = x, y
    for i in range(1, n):
        rule = get_rule()
        trans_mat = np.array([[rule['a'], rule['b'], rule['tx']],
                              [rule['c'], rule['d'], rule['ty']]])
        points[i, :2] = np.matmul(trans_mat, points[i - 1, :])
    return points


def main():
    n = 100000
    data = create_points(n)
    plt.scatter(data[:, 0], data[:, 1], color='blue', s=0.2)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()
