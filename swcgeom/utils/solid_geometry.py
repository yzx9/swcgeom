"""Solid Geometry."""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from sympy import Eq, solve, symbols

__all__ = [
    "find_unit_vector_on_plane",
    "find_sphere_line_intersection",
    "project_point_on_line",
]


def find_unit_vector_on_plane(normal_vec3: npt.NDArray) -> npt.NDArray:
    r = np.random.rand(3)
    r /= np.linalg.norm(r)
    while np.allclose(r, normal_vec3) or np.allclose(r, -normal_vec3):
        r = np.random.rand(3)
        r /= np.linalg.norm(r)

    u = np.cross(r, normal_vec3)
    u /= np.linalg.norm(u)
    return u


def find_sphere_line_intersection(
    sphere_center: npt.NDArray,
    sphere_radius: float,
    line_point_a: npt.NDArray,
    line_point_b: npt.NDArray,
) -> List[Tuple[float, npt.NDArray[np.float64]]]:
    x1, y1, z1 = sphere_center
    x2, y2, z2 = line_point_a
    x3, y3, z3 = line_point_b
    t = symbols("t", real=True)

    # line
    x = x2 + t * (x3 - x2)
    y = y2 + t * (y3 - y2)
    z = z2 + t * (z3 - z2)

    # sphere
    sphere_eq = Eq((x - x1) ** 2 + (y - y1) ** 2 + (z - z1) ** 2, sphere_radius**2)

    # solve
    t_values = solve(sphere_eq, t)
    intersections = [
        np.array([x.subs(t, t_val), y.subs(t, t_val), z.subs(t, t_val)], dtype=float)
        for t_val in t_values
    ]
    return list(zip(t_values, intersections))


def project_point_on_line(
    point_a: npt.ArrayLike, direction_vector: npt.ArrayLike, point_p: npt.ArrayLike
) -> npt.NDArray:
    A = np.array(point_a)
    n = np.array(direction_vector)
    P = np.array(point_p)

    AP = P - A
    projection = A + np.dot(AP, n) / np.dot(n, n) * n
    return projection
