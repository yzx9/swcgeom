"""Geometry object."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from swcgeom.utils.solid_geometry import (
    find_sphere_line_intersection,
    find_unit_vector_on_plane,
    project_point_on_line,
)

__all__ = ["GeomObject", "GeomSphere", "GeomFrustumCone"]

eps = 1e-6


class GeomObject(ABC):
    """Geometry object."""

    volume = None

    def get_volume(self) -> float:
        """Get volume."""
        if self.volume is None:
            self.volume = self._get_volume()
        return self.volume

    @abstractmethod
    def _get_volume(self) -> float:
        """Get volume."""
        raise NotImplementedError()

    @abstractmethod
    def union(self, obj: "GeomObject") -> "GeomObject":
        """Union with another geometry object."""
        classname = obj.__class__.__name__
        raise NotImplementedError(f"unable to union with {classname}")

    @abstractmethod
    def intersect(self, obj: "GeomObject") -> "GeomObject":
        """Intersect with another geometry object."""
        classname = obj.__class__.__name__
        raise NotImplementedError(f"unable to intersect with {classname}")


# Primitive Geometry Objects


class GeomSphere(GeomObject):
    """Geometry Sphere."""

    def __init__(self, center: npt.ArrayLike, radius: float):
        super().__init__()

        self.center = np.array(center)
        assert len(self.center) == 3

        self.radius = radius

    def _get_volume(self) -> float:
        return self.calc_volume(self.radius)

    def get_volume_spherical_cap(self, h: float) -> float:
        return self.calc_volume_spherical_cap(self.radius, h)

    def union(self, obj: GeomObject) -> GeomObject:
        if isinstance(obj, GeomSphere):
            return GeomSphere2Union(self, obj)

        if isinstance(obj, GeomFrustumCone):
            return GeomSphereFrustumConeUnion(self, obj)

        return super().union(obj)

    def intersect(self, obj: GeomObject) -> GeomObject:
        if isinstance(obj, GeomSphere):
            return GeomSphere2Intersection(self, obj)

        if isinstance(obj, GeomFrustumCone):
            return GeomSphereFrustumConeIntersection(self, obj)

        return super().intersect(obj)

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

    def _get_volume(self) -> float:
        return self.calc_volume(self.r1, self.r2, self.height())

    def union(self, obj: GeomObject) -> GeomObject:
        if isinstance(obj, GeomSphere):
            return GeomSphereFrustumConeUnion(obj, self)

        return super().union(obj)

    def intersect(self, obj: GeomObject) -> GeomObject:
        return super().intersect(obj)

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


# Composite Geometry Objects

T = TypeVar("T", bound=GeomObject)
K = TypeVar("K", bound=GeomObject)


class GeomComposition2(GeomObject, ABC, Generic[T, K]):
    """Composition of two geometry objects."""

    def __init__(self, obj1: T, obj2: K) -> None:
        super().__init__()
        self.obj1 = obj1
        self.obj2 = obj2

    def union(self, obj: GeomObject) -> GeomObject:
        return super().union(obj)

    def intersect(self, obj: GeomObject) -> GeomObject:
        return super().intersect(obj)

    def get_union_volume(self) -> float:
        return (
            self.obj1.get_volume()
            + self.obj2.get_volume()
            - self.get_intersect_volume()
        )

    @abstractmethod
    def get_intersect_volume(self) -> float:
        raise NotImplementedError()


# Composite sphere and sphere


class GeomSphere2Composition(GeomComposition2[GeomSphere, GeomSphere]):
    def get_intersect_volume(self) -> float:
        return self.calc_intersect_volume(self.obj1, self.obj2)

    @staticmethod
    def calc_intersect_volume(obj1: GeomSphere, obj2: GeomSphere) -> float:
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
            return GeomSphere.calc_volume(min(r1, r2))

        part1 = (np.pi / (12 * d)) * (r1 + r2 - d) ** 2
        part2 = (
            d**2 + 2 * d * r1 - 3 * r1**2 + 2 * d * r2 - 3 * r2**2 + 6 * r1 * r2
        )
        return part1 * part2


class GeomSphere2Union(GeomSphere2Composition):
    """Union of two spheres."""

    def _get_volume(self) -> float:
        return self.get_union_volume()


class GeomSphere2Intersection(GeomSphere2Composition):
    """Intersection of two spheres."""

    def _get_volume(self) -> float:
        return self.get_intersect_volume()


# Composite sphere and frustum cone


class GeomSphereFrustumConeComposite(GeomComposition2[GeomSphere, GeomFrustumCone]):
    def get_intersect_volume(self) -> float:
        return self.calc_intersect_volume(self.obj1, self.obj2)

    @staticmethod
    def calc_intersect_volume(
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
        v = find_unit_vector_on_plane(up)

        intersections = find_sphere_line_intersection(c1, r1, c1 + r1 * v, c2 + r2 * v)
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

        M = project_point_on_line(c1, up, p)
        h1 = np.linalg.norm(M - c1).item()
        r3 = np.linalg.norm(M - p).item()
        v_cap1 = GeomSphere.calc_volume_spherical_cap(r1, r1 - h1)
        v_frustum = GeomFrustumCone.calc_volume(r1, r3, h1)

        # Fast-Path: The frustum cone is higher than the sphere
        if h >= r1:
            return v_cap1 + v_frustum

        v_cap2 = GeomSphere.calc_volume_spherical_cap(r1, r1 - h)
        return v_cap1 + v_frustum - v_cap2


class GeomSphereFrustumConeUnion(GeomSphereFrustumConeComposite):
    """Union of sphere and frustum cone."""

    def _get_volume(self) -> float:
        return self.get_union_volume()


class GeomSphereFrustumConeIntersection(GeomSphereFrustumConeComposite):
    """Intersection of sphere and frustum cone."""

    def _get_volume(self) -> float:
        return self.get_intersect_volume()
