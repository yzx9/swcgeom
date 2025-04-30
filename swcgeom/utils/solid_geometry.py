# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Solid Geometry."""

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
    """Find a random unit vector on the plane defined by the normal vector.

    >>> normal = np.array([0, 0, 1])
    >>> u = find_unit_vector_on_plane(normal)
    >>> np.allclose(np.dot(u, normal), 0)  # Should be perpendicular
    True
    >>> np.allclose(np.linalg.norm(u), 1)  # Should be unit length
    True
    """
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
) -> list[tuple[float, npt.NDArray[np.float64]]]:
    """Find intersection points between a sphere and a line.

    >>> center = np.array([0, 0, 0])
    >>> radius = 1.0
    >>> p1 = np.array([-2, 0, 0])
    >>> p2 = np.array([2, 0, 0])
    >>> intersections = find_sphere_line_intersection(center, radius, p1, p2)
    >>> len(intersections)
    2
    >>> np.allclose(intersections[0][1], [-1, 0, 0])
    True
    >>> np.allclose(intersections[1][1], [1, 0, 0])
    True
    """

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
    point_a: npt.ArrayLike,
    direction_vector: npt.ArrayLike,
    point_p: npt.ArrayLike,
) -> npt.NDArray:
    """Project a point onto a line defined by a point and direction vector.

    >>> a = np.array([0, 0, 0])
    >>> d = np.array([1, 0, 0])
    >>> p = np.array([1, 1, 0])
    >>> projection = project_point_on_line(a, d, p)
    >>> np.allclose(projection, [1, 0, 0])
    True
    """

    A = np.array(point_a)
    n = np.array(direction_vector)
    P = np.array(point_p)

    AP = P - A
    projection = A + np.dot(AP, n) / np.dot(n, n) * n
    return projection


def project_vector_on_vector(vec: npt.ArrayLike, target: npt.ArrayLike) -> npt.NDArray:
    """Project one vector onto another.

    >>> v = np.array([1, 1, 0])
    >>> t = np.array([1, 0, 0])
    >>> proj = project_vector_on_vector(v, t)
    >>> np.allclose(proj, [1, 0, 0])
    True
    """
    v = np.array(vec)
    n = np.array(target)

    n_normalized = n / np.linalg.norm(n)
    projection_on_n = np.dot(v, n_normalized) * n_normalized
    return projection_on_n


def project_vector_on_plane(
    vec: npt.ArrayLike,
    plane_normal_vec: npt.ArrayLike,
) -> npt.NDArray:
    """Project a vector onto a plane defined by its normal vector.

    >>> v = np.array([1, 1, 1])
    >>> n = np.array([0, 0, 1])
    >>> proj = project_vector_on_plane(v, n)
    >>> np.allclose(proj, [1, 1, 0])  # Z component removed
    True
    >>> np.allclose(np.dot(proj, n), 0)  # Should be perpendicular to normal
    True
    """

    v = np.array(vec)
    n = np.array(plane_normal_vec)

    # project v to n
    projection_on_n = project_vector_on_vector(vec, n)

    # project v to plane
    projection_on_plane = v - projection_on_n

    return projection_on_plane
