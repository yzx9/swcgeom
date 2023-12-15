"""Geometry object."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from sympy import Eq, solve, symbols

__all__ = ["GeomObject", "GeomSphere", "GeomFrustumCone"]

eps = 1e-6


class GeomObject(ABC):
    """Geometry object."""

    @abstractmethod
    def get_volume(self) -> float:
        """Get volume."""
        raise NotImplementedError()

    @abstractmethod
    def get_intersect_volume(self, obj: "GeomObject") -> float:
        """Get intersect volume.

        Parameters
        ----------
        obj : GeometryObject
            Another geometry object.

        Returns
        -------
        volume : float
            Intersect volume.
        """
        raise NotImplementedError()


class GeomSphere(GeomObject):
    """Geometry Sphere."""

    def __init__(self, center: npt.ArrayLike, radius: float):
        super().__init__()

        self.center = np.array(center)
        assert len(self.center) == 3

        self.radius = radius

    def get_volume(self) -> float:
        return self.calc_volume(self.radius)

    def get_volume_spherical_cap(self, h: float) -> float:
        return self.calc_volume_spherical_cap(self.radius, h)

    def get_intersect_volume_sphere(self, obj: "GeomSphere") -> float:
        return self.calc_intersect_volume_sphere(self, obj)

    def get_intersect_volume_sphere_frustum_cone(
        self, frustum_cone: "GeomFrustumCone"
    ) -> float:
        return calc_intersect_volume_sphere_frustum_cone(self, frustum_cone)

    def get_intersect_volume(self, obj: GeomObject) -> float:
        if isinstance(obj, GeomSphere):
            return self.get_intersect_volume_sphere(obj)

        if isinstance(obj, GeomFrustumCone):
            return self.get_intersect_volume_sphere_frustum_cone(obj)

        classname = obj.__class__.__name__
        raise NotImplementedError(f"unsupported geometry object: {classname}")

    @staticmethod
    def calc_volume(radius: float) -> float:
        r"""Calculate volume of sphere.

        \being{equation}
        V = \frac{4}{3} * π * r^3
        \end{equation}

        Returns
        -------
        volume : float
            Volume.
        """
        return 4 / 3 * np.pi * radius**3

    @staticmethod
    def calc_volume_spherical_cap(r: float, h: float) -> float:
        r"""Calculate the volume of a spherical cap.

        \being{equation}
        V = π * h^2 * (3r - h) / 3
        \end{equation}

        Parameters
        ----------
        r : float
            radius of the sphere
        h : float
            height of the spherical cap

        Returns
        -------
        volume : float
            volume of the spherical cap
        """
        return np.pi * h**2 * (3 * r - h) / 3

    @classmethod
    def calc_intersect_volume_sphere(
        cls, obj1: "GeomSphere", obj2: "GeomSphere"
    ) -> float:
        r"""Calculate intersect volume of two spheres.

        \being{equation}
        V = \frac{\pi}{12d} * (r_1 + r_2 - d)^2 (d^2 + 2d r_1 - 3r_1^2 + 2d r_2 - 3r_2^2 + 6 r_1r_2)
        \end{equation}

        Returns
        -------
        volume : float
            Intersect volume.
        """

        r1, r2 = obj1.radius, obj2.radius
        d = np.linalg.norm(obj1.center - obj2.center).item()
        if d > r1 + r2:
            return 0

        if d <= abs(r1 - r2):
            return cls.calc_volume(min(r1, r2))

        part1 = (np.pi / (12 * d)) * (r1 + r2 - d) ** 2
        part2 = (
            d**2 + 2 * d * r1 - 3 * r1**2 + 2 * d * r2 - 3 * r2**2 + 6 * r1 * r2
        )
        return part1 * part2


class GeomFrustumCone(GeomObject):
    """Geometry Frustum."""

    def __init__(self, c1: npt.ArrayLike, r1: float, c2: npt.ArrayLike, r2: float):
        super().__init__()

        self.c1 = np.array(c1)
        assert len(self.c1) == 3

        self.c2 = np.array(c2)
        assert len(self.c2) == 3

        self.r1 = r1
        self.r2 = r2

    def height(self) -> float:
        """Get height of frustum."""
        return np.linalg.norm(self.c1 - self.c2).item()

    def get_volume(self) -> float:
        return self.calc_volume(self.r1, self.r2, self.height())

    def get_intersect_volume_sphere(self, sphere: GeomSphere) -> float:
        return calc_intersect_volume_sphere_frustum_cone(sphere, self)

    def get_intersect_volume(self, obj: GeomObject) -> float:
        if isinstance(obj, GeomSphere):
            return self.get_intersect_volume_sphere(obj)

        classname = obj.__class__.__name__
        raise NotImplementedError(f"unsupported geometry object: {classname}")

    @staticmethod
    def calc_volume(r1: float, r2: float, height: float) -> float:
        r"""Calculate volume of frustum.

        \being{equation}
        V = \frac{1}{3} * π * h * (r^2 + r * R + R^2)
        \end{equation}

        Returns
        -------
        volume : float
            Volume.
        """
        return (1 / 3) * np.pi * height * (r1**2 + r1 * r2 + r2**2)


@lru_cache
def calc_intersect_volume_sphere_frustum_cone(
    sphere: GeomSphere, frustum_cone: GeomFrustumCone
) -> float:
    r"""Calculate intersect volume of sphere and frustum cone.

    Returns
    -------
    volume : float
        Intersect volume.
    """
    h = frustum_cone.height()
    c1, r1 = sphere.center, sphere.radius
    if np.allclose(c1, frustum_cone.c1) and np.allclose(r1, frustum_cone.r1):
        c2, r2 = frustum_cone.c2, frustum_cone.r2
    elif np.allclose(c1, frustum_cone.c2) and np.allclose(r1, frustum_cone.r2):
        c2, r2 = frustum_cone.c1, frustum_cone.r1
    else:
        raise NotImplementedError("unsupported to calculate intersect volume")

    # Fast-Path: The surface of the frustum concentric with the sphere
    # is the surface with smaller radius
    if r2 - r1 >= -eps:  # r2 >= r1:
        v_himisphere = GeomSphere.calc_volume_spherical_cap(r1, r1)
        if h >= r1:
            # The hemisphere is completely inside the frustum cone
            return v_himisphere

        # The frustum cone is lower than the hemisphere
        v_cap = GeomSphere.calc_volume_spherical_cap(r1, r1 - h)
        return v_himisphere - v_cap

    up = (c2 - c1) / np.linalg.norm(c2 - c1)
    v = _find_unit_vector_on_plane(up)

    intersections = _find_sphere_line_intersection(c1, r1, c1 + r1 * v, c2 + r2 * v)
    if len(intersections) == 0:
        # Tricky case: Since the intersection point not found with
        # numerical precision, we can simply assume that there are two
        # intersection points and at the same position
        intersections = [(0, c1 + r1 * v), (0, c1 + r1 * v)]
    assert len(intersections) == 2
    t, p = max(intersections, key=lambda x: x[0])

    # Fast-Path: The frustum cone is completely inside the sphere
    if t > 1 + eps:
        return frustum_cone.get_volume()

    M = _project_point_on_line(c1, up, p)
    h1 = np.linalg.norm(M - c1).item()
    r3 = np.linalg.norm(M - p).item()
    v_cap1 = GeomSphere.calc_volume_spherical_cap(r1, r1 - h1)
    v_frustum = GeomFrustumCone.calc_volume(r1, r3, h1)

    # Fast-Path: The frustum cone is higher than the sphere
    if h >= r1:
        return v_cap1 + v_frustum

    v_cap2 = GeomSphere.calc_volume_spherical_cap(r1, r1 - h)
    return v_cap1 + v_frustum - v_cap2


def _find_unit_vector_on_plane(normal_vec3: npt.NDArray) -> npt.NDArray:
    r = np.random.rand(3)
    r /= np.linalg.norm(r)
    while np.allclose(r, normal_vec3) or np.allclose(r, -normal_vec3):
        r = np.random.rand(3)
        r /= np.linalg.norm(r)

    u = np.cross(r, normal_vec3)
    u /= np.linalg.norm(u)
    return u


def _find_sphere_line_intersection(
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


def _project_point_on_line(
    point_a: npt.ArrayLike, direction_vector: npt.ArrayLike, point_p: npt.ArrayLike
) -> npt.NDArray:
    A = np.array(point_a)
    n = np.array(direction_vector)
    P = np.array(point_p)

    AP = P - A
    projection = A + np.dot(AP, n) / np.dot(n, n) * n
    return projection


if __name__ == "__main__":
    sphere = GeomSphere((391.58, 324.97, -12.89), 0.493507)
    frustum_cone = GeomFrustumCone(
        (391.58, 324.97, -12.89), 0.493507, (388.07, 320.41, -13.57), 0.493506
    )
    print(calc_intersect_volume_sphere_frustum_cone(sphere, frustum_cone))
