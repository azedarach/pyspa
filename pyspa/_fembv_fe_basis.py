from __future__ import division

import numpy as np


class PiecewiseConstantFEMBasis(object):
    def __init__(self, width=1, value=1):
        self.width = width
        self.value = value

    def _width(self):
        width = int(np.floor(self.width))
        if width < 1:
            width == 1
        return width

    def _basis_func(self, i, grid_points):
        value = np.zeros(grid_points.shape)
        value[grid_points - i < self._width()] = self.value
        value[grid_points < i] = 0
        return value

    def __call__(self, n_grid_points):
        grid_points = np.arange(n_grid_points)
        width = self._width()

        i_max = int(np.floor((n_grid_points - 1) / width))
        left_end_points = width * np.arange(i_max + 1)

        n_elements = np.size(left_end_points)
        V = np.zeros((n_grid_points, n_elements))
        for i, v in enumerate(left_end_points):
            V[:, i] = self._basis_func(v, grid_points)

        return V


class TriangleFEMBasis(object):
    def __init__(self, n_points=3, value=1):
        self.n_points = n_points
        self.value = value

    def _basis_func(self, i, grid_points):
        half_width = int((self.n_points - 1) / 2)
        slope = self.value / half_width
        value = np.zeros(grid_points.shape)
        value[grid_points <= i] = (
            self.value + slope * (grid_points[grid_points <= i] - i))
        value[grid_points > i] = (
            self.value - slope * (grid_points[grid_points > i] - i))
        value[np.abs(grid_points - i) >= half_width] = 0
        return value

    def __call__(self, n_grid_points):
        if self.n_points % 2 == 0:
            raise ValueError('number of element points must be an odd number')

        half_width = int((self.n_points - 1) / 2)
        if n_grid_points < half_width:
            raise ValueError('too few grid points')

        grid_points = np.arange(n_grid_points)
        i_max = int(np.floor((n_grid_points - 1) / half_width))
        midpoints = half_width * np.arange(i_max + 1)
        n_elements = np.size(midpoints)
        V = np.zeros((n_grid_points, n_elements))
        for i, v in enumerate(midpoints):
            V[:, i] = self._basis_func(v, grid_points)

        return V
