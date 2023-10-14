"""Finds the Minimum Volume Enclosing Ellipsoid."""

# pylint: disable=invalid-name

from functools import cached_property
from typing import Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt

__all__ = ["mvee"]


class Ellipse:
    """Ellipse wrapper."""

    def __init__(
        self, A: npt.NDArray[np.float64], centroid: npt.NDArray[np.float64]
    ) -> None:
        self.A = A
        self.centroid = centroid

    @property
    def radii(self) -> Tuple[float, float]:
        # x, y radii.
        _U, D, _V = self.svd
        rx, ry = 1.0 / np.sqrt(D)
        return rx, ry

    @property
    def a(self) -> float:
        a, _b = self.axes
        return a

    @property
    def b(self) -> float:
        _a, b = self.axes
        return b

    @property
    def axes(self) -> Tuple[float, float]:
        # Major and minor semi-axis of the ellipse.
        rx, ry = self.radii
        dx, dy = 2 * rx, 2 * ry
        a, b = max(dx, dy), min(dx, dy)
        return a, b

    @property
    def eccentricity(self) -> float:
        a, b = self.axes
        e = np.sqrt(a**2 - b**2) / a
        return e

    @property
    def alpha(self) -> float:
        # Orientation angle (with respect to the x axis counterclockwise).
        _U, _D, V = self.svd
        arcsin = -1.0 * np.rad2deg(np.arcsin(V[0][0]))
        arccos = np.rad2deg(np.arccos(V[0][1]))
        alpha = arccos if arcsin > 0.0 else -1.0 * arccos
        return alpha

    @cached_property
    def svd(self):
        # V is the rotation matrix that gives the orientation of the ellipsoid.
        # https://en.wikipedia.org/wiki/Rotation_matrix
        # http://mathworld.wolfram.com/RotationMatrix.html
        U, D, V = la.svd(self.A)
        return U, D, V


def mvee(points: npt.NDArray[np.floating], tol: float = 1e-3) -> Ellipse:
    A, centroid = _mvee(points, tol=tol)
    return Ellipse(A, centroid)


def _mvee(
    points: npt.NDArray[np.floating], tol: float = 1e-3
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Finds the Minimum Volume Enclosing Ellipsoid.

    Returns
    -------
    A : matrix of shape (d, d)
        The ellipse equation in the 'center form': (x-c)' * A * (x-c) = 1
    centroid : array of shape (d,)
        The center coordinates of the ellipse.

    Reference
    ---------
    1. http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
    2. http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440
    3. https://minillinim.github.io/GroopM/dev_docs/groopm.ellipsoid-pysrc.html
    """

    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u
    c = np.dot(u, points)
    A = (
        la.inv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(c, c))
        / d
    )
    return A, c
