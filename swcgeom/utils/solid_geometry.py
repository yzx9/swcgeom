"""Solid Geometry."""

from typing import List, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "find_unit_vector_on_plane",
    "find_sphere_line_intersection",
    "project_point_on_line",
    "project_vector_on_vector",
    "project_vector_on_plane",
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
    A = np.array(line_point_a)
    B = np.array(line_point_b)
    C = np.array(sphere_center)

    D = B - A  # line direction vector
    f = A - C  # line to sphere center

    # solve: a*t^2 + b*t + c = 0
    a = np.dot(D, D)
    b = 2 * np.dot(f, D)
    c = np.dot(f, f) - sphere_radius**2
    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []  # no intersection

    if discriminant == 0:
        t = -b / (2 * a)
        p = A + t * D
        return [(t, p)]  # single intersection, a tangent

    t1 = (-b - np.sqrt(discriminant)) / (2 * a)
    t2 = (-b + np.sqrt(discriminant)) / (2 * a)
    p1 = A + t1 * D
    p2 = A + t2 * D
    return [(t1, p1), (t2, p2)]


def project_point_on_line(
    point_a: npt.ArrayLike, direction_vector: npt.ArrayLike, point_p: npt.ArrayLike
) -> npt.NDArray:
    A = np.array(point_a)
    n = np.array(direction_vector)
    P = np.array(point_p)

    AP = P - A
    projection = A + np.dot(AP, n) / np.dot(n, n) * n
    return projection


def project_vector_on_vector(vec: npt.ArrayLike, target: npt.ArrayLike) -> npt.NDArray:
    v = np.array(vec)
    n = np.array(target)

    n_normalized = n / np.linalg.norm(n)
    projection_on_n = np.dot(v, n_normalized) * n_normalized
    return projection_on_n


def project_vector_on_plane(
    vec: npt.ArrayLike, plane_normal_vec: npt.ArrayLike
) -> npt.NDArray:
    v = np.array(vec)
    n = np.array(plane_normal_vec)

    # project v to n
    projection_on_n = project_vector_on_vector(vec, n)

    # project v to plane
    projection_on_plane = v - projection_on_n

    return projection_on_plane
