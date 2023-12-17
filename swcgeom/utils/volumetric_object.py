"""Volumetric object.

This library implements the calculation of volumes for any shape
generated through boolean operations, employing Signed Distance
Function (SDF) and Monte Carlo algorithms.

However, this approach is computationally demanding. To address this,
we have specialized certain operations to accelerate the computation
process.

If you wish to use these methods, please review our implementation.
Additionally, consider specializing some subclasses that can utilize
formula-based calculations for further optimization of your
computations.
"""

from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from swcgeom.utils.sdf import (
    AABB,
    SDF,
    SDFDifference,
    SDFFrustumCone,
    SDFIntersection,
    SDFSphere,
    SDFUnion,
)
from swcgeom.utils.solid_geometry import (
    find_sphere_line_intersection,
    find_unit_vector_on_plane,
    project_point_on_line,
)

__all__ = ["VolObject", "VolMCObject", "VolSDFObject", "VolSphere", "VolFrustumCone"]

eps = 1e-6


class VolObject(ABC):
    """Volumetric object."""

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
    def union(self, obj: "VolObject") -> "VolObject":
        """Union with another volume object."""
        classname = obj.__class__.__name__
        raise NotImplementedError(f"unable to union with {classname}")

    @abstractmethod
    def intersect(self, obj: "VolObject") -> "VolObject":
        """Intersect with another volume object."""
        classname = obj.__class__.__name__
        raise NotImplementedError(f"unable to intersect with {classname}")

    @abstractmethod
    def subtract(self, obj: "VolObject") -> "VolObject":
        """Subtract another volume object."""
        classname = obj.__class__.__name__
        raise NotImplementedError(f"unable to diff with {classname}")


class VolMCObject(VolObject, ABC):
    """Volumetric Monte Carlo Object.

    The volume of the object is calculated by Monte Carlo integration.
    """

    def __init__(self, *, n_samples: int = 1000000) -> None:
        super().__init__()
        self.n_samples = n_samples

    @abstractmethod
    def sample(self, n: int) -> Tuple[npt.NDArray[np.float32], float]:
        """Sample points.

        Parameters
        ----------
        n : int
            Number of points to sample.

        Returns
        -------
        points : ndarray
            Sampled points.
        volume : float
            Volume of the sample range.
        """
        raise NotImplementedError()

    @abstractmethod
    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Is p in the object.

        Returns
        -------
        is_in : npt.NDArray[np.bool_]
            Array of shape (N,), if bounding box is `None`, `True` will
            be returned.
        """
        raise NotImplementedError()

    def _get_volume(self) -> float:
        p, v = self.sample(self.n_samples)
        insides = self.is_in(p)
        hits = np.count_nonzero(insides)
        return hits / self.n_samples * v


# Volumetric SDF Objects


class VolSDFObject(VolMCObject):
    """Volumetric SDF Object.

    Notes
    -----
    SDF must has a bounding box.
    """

    def __init__(self, sdf: SDF, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sdf = sdf
        assert self.sdf.bounding_box is not None

    def sample(self, n: int) -> Tuple[npt.NDArray[np.float32], float]:
        (min_x, min_y, min_z), (max_x, max_y, max_z) = self.get_bounding_box()
        samples = np.random.uniform(
            (min_x, min_y, min_z), (max_x, max_y, max_z), size=(n, 3)
        ).astype(np.float32)
        v = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        return samples, v

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        return self.sdf.is_in(p)

    def get_bounding_box(self) -> AABB:
        assert self.sdf.bounding_box is not None
        return self.sdf.bounding_box

    def union(self, obj: VolObject) -> VolObject:
        if isinstance(obj, VolSDFObject):
            return VolSDFUnion(self, obj)
        return super().union(obj)

    def intersect(self, obj: VolObject) -> VolObject:
        if isinstance(obj, VolSDFObject):
            return VolSDFIntersection(self, obj)
        return super().intersect(obj)

    def subtract(self, obj: VolObject) -> VolObject:
        if isinstance(obj, VolSDFObject):
            return VolSDFDifference(self, obj)
        return super().subtract(obj)


T = TypeVar("T", bound=VolSDFObject)
K = TypeVar("K", bound=VolSDFObject)


class VolSDFIntersection(VolSDFObject, ABC, Generic[T, K]):
    """Intersection of two volumetric sdf objects."""

    def __init__(self, obj1: T, obj2: K, **kwargs) -> None:
        obj = SDFIntersection(obj1.sdf, obj2.sdf)
        super().__init__(obj, **kwargs)
        self.obj1 = obj1
        self.obj2 = obj2


class VolSDFUnion(VolSDFObject, ABC, Generic[T, K]):
    """Union of two volumetric sdf objects."""

    def __init__(self, obj1: T, obj2: K, **kwargs) -> None:
        obj = SDFUnion(obj1.sdf, obj2.sdf)
        super().__init__(obj, **kwargs)
        self.obj1 = obj1
        self.obj2 = obj2


class VolSDFDifference(VolSDFObject, ABC, Generic[T, K]):
    """Difference of volumetric sdf object and another object."""

    def __init__(self, obj1: T, obj2: K, **kwargs) -> None:
        obj = SDFDifference(obj1.sdf, obj2.sdf)
        super().__init__(obj, **kwargs)
        self.obj1 = obj1
        self.obj2 = obj2


# Primitive Volumetric Objects


class VolSphere(VolSDFObject):
    """Volumetric Sphere."""

    def __init__(self, center: npt.ArrayLike, radius: float):
        center = np.array(center)
        assert len(center) == 3

        sdf = SDFSphere(center, radius)
        super().__init__(sdf)

        self.center = center
        self.radius = radius

    def _get_volume(self) -> float:
        return self.calc_volume(self.radius)

    def get_volume_spherical_cap(self, h: float) -> float:
        return self.calc_volume_spherical_cap(self.radius, h)

    def union(self, obj: VolObject) -> VolObject:
        if isinstance(obj, VolSphere):
            return VolSphere2Union(self, obj)

        if isinstance(obj, VolFrustumCone):
            return VolSphereFrustumConeUnion(self, obj)

        return super().union(obj)

    def intersect(self, obj: VolObject) -> VolObject:
        if isinstance(obj, VolSphere):
            return VolSphere2Intersection(self, obj)

        if isinstance(obj, VolFrustumCone):
            return VolSphereFrustumConeIntersection(self, obj)

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


class VolFrustumCone(VolSDFObject):
    """Volumetric Frustum."""

    def __init__(self, c1: npt.ArrayLike, r1: float, c2: npt.ArrayLike, r2: float):
        c1 = np.array(c1)
        assert len(c1) == 3

        c2 = np.array(c2)
        assert len(c2) == 3

        sdf = SDFFrustumCone(c1, c2, r1, r2)
        super().__init__(sdf)

        self.c1 = c1
        self.c2 = c2
        self.r1 = r1
        self.r2 = r2

    def height(self) -> float:
        """Get height of frustum."""
        return np.linalg.norm(self.c1 - self.c2).item()

    def _get_volume(self) -> float:
        return self.calc_volume(self.r1, self.r2, self.height())

    def union(self, obj: VolObject) -> VolObject:
        if isinstance(obj, VolSphere):
            return VolSphereFrustumConeUnion(obj, self)

        return super().union(obj)

    def intersect(self, obj: VolObject) -> VolObject:
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


# Composite sphere and sphere


class VolSphere2Intersection(VolSDFIntersection[VolSphere, VolSphere]):
    """Intersection of two spheres."""

    def _get_volume(self) -> float:
        return self.calc_intersect_volume(self.obj1, self.obj2)

    @staticmethod
    def calc_intersect_volume(obj1: VolSphere, obj2: VolSphere) -> float:
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
            return VolSphere.calc_volume(min(r1, r2))

        part1 = (np.pi / (12 * d)) * (r1 + r2 - d) ** 2
        part2 = (
            d**2 + 2 * d * r1 - 3 * r1**2 + 2 * d * r2 - 3 * r2**2 + 6 * r1 * r2
        )
        return part1 * part2


class VolSphere2Union(VolSDFUnion[VolSphere, VolSphere]):
    """Union of two spheres."""

    def _get_volume(self) -> float:
        return (
            self.obj1.get_volume()
            + self.obj2.get_volume()
            - VolSphere2Intersection.calc_intersect_volume(self.obj1, self.obj2)
        )


# Composite sphere and frustum cone


class VolSphereFrustumConeIntersection(VolSDFIntersection[VolSphere, VolFrustumCone]):
    """Intersection of sphere and frustum cone."""

    def _get_volume(self) -> float:
        if (
            np.allclose(self.obj1.center, self.obj2.c1)
            and np.allclose(self.obj1.radius, self.obj2.r1)
        ) or (
            np.allclose(self.obj1.center, self.obj2.c2)
            and np.allclose(self.obj1.radius, self.obj2.r2)
        ):
            return self.calc_concentric_intersect_volume(self.obj1, self.obj2)

        return super()._get_volume()

    @staticmethod
    def calc_concentric_intersect_volume(
        sphere: VolSphere, frustum_cone: VolFrustumCone
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
            raise ValueError("sphere and frustum cone is not concentric")

        # Fast-Path: The surface of the frustum concentric with the sphere
        # is the surface with smaller radius
        if r2 - r1 >= -eps:  # r2 >= r1:
            v_himisphere = VolSphere.calc_volume_spherical_cap(r1, r1)
            if h >= r1:
                # The hemisphere is completely inside the frustum cone
                return v_himisphere

            # The frustum cone is lower than the hemisphere
            v_cap = VolSphere.calc_volume_spherical_cap(r1, r1 - h)
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
        v_cap1 = VolSphere.calc_volume_spherical_cap(r1, r1 - h1)
        v_frustum = VolFrustumCone.calc_volume(r1, r3, h1)

        # Fast-Path: The frustum cone is higher than the sphere
        if h >= r1:
            return v_cap1 + v_frustum

        v_cap2 = VolSphere.calc_volume_spherical_cap(r1, r1 - h)
        return v_cap1 + v_frustum - v_cap2


class VolSphereFrustumConeUnion(VolSDFUnion[VolSphere, VolFrustumCone]):
    """Union of sphere and frustum cone."""

    def _get_volume(self) -> float:
        return (
            self.obj1.get_volume()
            + self.obj2.get_volume()
            - VolSphereFrustumConeIntersection.calc_concentric_intersect_volume(
                self.obj1, self.obj2
            )
        )
