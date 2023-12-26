"""Signed distance functions.

Refs: https://iquilezles.org/articles/distfunctions/

Note
----
This module has been deprecated since v0.14.0, and will be removed in
the future, use `sdflit` instead.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

import numpy as np
import numpy.typing as npt

from swcgeom.utils.solid_geometry import project_vector_on_plane

__all__ = [
    "SDF",
    "SDFUnion",
    "SDFIntersection",
    "SDFDifference",
    "SDFCompose",
    "SDFSphere",
    "SDFFrustumCone",
    "SDFRoundCone",
]

# Axis-aligned bounding box, tuple of array of shape (3,)
AABB = Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]


class SDF(ABC):
    """Signed distance functions."""

    bounding_box: AABB | None = None

    def __call__(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self.distance(p)

    @abstractmethod
    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calculate signed distance.

        Parmeters
        ---------
        p: ArrayLike
            Hit point p of shape (N, 3).

        Returns
        -------
        distance : npt.NDArray[np.float32]
            Distance array of shape (3,).
        """
        raise NotImplementedError()

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        in_box = self.is_in_bounding_box(p)
        flags = np.full((p.shape[0]), False, dtype=np.bool_)
        flags[in_box] = self.distance(p[in_box]) <= 0
        return flags

    def is_in_bounding_box(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        """Is p in bounding box.

        Returns
        -------
        is_in : npt.NDArray[np.bool_]
            Array of shape (N,), if bounding box is `None`, `True` will
            be returned.
        """

        if self.bounding_box is None:
            return np.full((p.shape[0]), True, dtype=np.bool_)

        is_in = np.logical_and(
            np.all(p >= self.bounding_box[0], axis=1),
            np.all(p <= self.bounding_box[1], axis=1),
        )
        return is_in


class SDFUnion(SDF):
    """Union multiple SDFs."""

    def __init__(self, *sdfs: SDF) -> None:
        assert len(sdfs) != 0, "must combine at least one SDF"
        super().__init__()

        self.sdfs = sdfs

        bounding_boxes = [sdf.bounding_box for sdf in sdfs if sdf.bounding_box]
        if len(bounding_boxes) == len(self.sdfs):
            self.bounding_box = (
                np.min(np.stack([box[0] for box in bounding_boxes]).T, axis=1),
                np.max(np.stack([box[1] for box in bounding_boxes]).T, axis=1),
            )

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.min([sdf(p) for sdf in self.sdfs], axis=0)

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        in_box = self.is_in_bounding_box(p)
        p_in_box = p[in_box]
        is_in = np.stack([sdf.is_in(p_in_box) for sdf in self.sdfs])
        flags = np.full_like(in_box, False, dtype=np.bool_)
        flags[in_box] = np.any(is_in, axis=0)
        return flags


class SDFIntersection(SDF):
    def __init__(self, *sdfs: SDF) -> None:
        assert len(sdfs) != 0, "must intersect at least one SDF"
        super().__init__()
        self.sdfs = sdfs

        bounding_boxes = [sdf.bounding_box for sdf in self.sdfs if sdf.bounding_box]
        if len(bounding_boxes) == len(self.sdfs):
            self.bounding_box = (
                np.max(np.stack([box[0] for box in bounding_boxes]).T, axis=1),
                np.min(np.stack([box[1] for box in bounding_boxes]).T, axis=1),
            )

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        distances = np.stack([sdf.distance(p) for sdf in self.sdfs])
        return np.max(distances, axis=1)

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        in_box = self.is_in_bounding_box(p)
        p_in_box = p[in_box]
        is_in = np.stack([sdf.is_in(p_in_box) for sdf in self.sdfs])
        flags = np.full_like(in_box, False, dtype=np.bool_)
        flags[in_box] = np.all(is_in, axis=0)
        return flags


class SDFDifference(SDF):
    """Difference of two SDFs A-B."""

    def __init__(self, sdf_a: SDF, sdf_b: SDF) -> None:
        super().__init__()
        self.sdf_a = sdf_a
        self.sdf_b = sdf_b

        self.bounding_box = sdf_a.bounding_box

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        da = self.sdf_a.distance(p)
        db = self.sdf_b.distance(p)
        return np.maximum(da, -db)

    def is_in(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        in_box = self.is_in_bounding_box(p)
        p_in_box = p[in_box]
        is_in_a = self.sdf_a.is_in(p_in_box)
        is_in_b = self.sdf_b.is_in(p_in_box)
        flags = np.full_like(in_box, False, dtype=np.bool_)
        flags[in_box] = np.logical_and(is_in_a, np.logical_not(is_in_b))
        return flags


class SDFCompose(SDFUnion):
    """Compose multiple SDFs."""

    def __init__(self, sdfs: Iterable[SDF]) -> None:
        warnings.warn(
            "`SDFCompose` has been replace by `SDFUnion` since v0.14.0, "
            "and will be removed in next version",
            DeprecationWarning,
        )
        sdfs = list(sdfs)
        if len(sdfs) == 1:
            warnings.warn("compose only one SDF, use SDFCompose.compose instead")

        super().__init__(*sdfs)

    @staticmethod
    def compose(sdfs: Iterable[SDF]) -> SDF:
        sdfs = list(sdfs)
        return SDFCompose(sdfs) if len(sdfs) != 1 else sdfs[0]


class SDFSphere(SDF):
    """SDF of sphere."""

    def __init__(self, center: npt.ArrayLike, radius: float) -> None:
        super().__init__()
        self.center = np.array(center)
        self.radius = radius
        assert tuple(self.center.shape) == (3,), "center should be vector of 3d"

        self.bounding_box = (self.center - self.radius, self.center + self.radius)

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.linalg.norm(p - self.center, axis=1) - self.radius


class SDFFrustumCone(SDF):
    """SDF of frustum cone."""

    def __init__(
        self, a: npt.ArrayLike, b: npt.ArrayLike, ra: float, rb: float
    ) -> None:
        super().__init__()
        self.a = np.array(a)
        self.b = np.array(b)
        self.ra = ra
        self.rb = rb
        assert tuple(self.a.shape) == (3,), "a should be vector of 3d"
        assert tuple(self.b.shape) == (3,), "b should be vector of 3d"

        self.bounding_box = self.get_bounding_box()

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        a, b, ra, rb = self.a, self.b, self.ra, self.rb

        rba = rb - ra
        baba = np.dot(b - a, b - a)
        papa = np.einsum("ij,ij->i", p - a, p - a)
        paba = np.dot(p - a, b - a) / baba
        # maybe negative due to numerical error
        x = np.sqrt(np.maximum(papa - paba * paba * baba, 0))
        cax = np.maximum(0.0, x - np.where(paba < 0.5, ra, rb))
        cay = np.abs(paba - 0.5) - 0.5
        k = rba * rba + baba
        f = np.clip((rba * (x - ra) + paba * baba) / k, 0.0, 1.0)
        cbx = x - ra - f * rba
        cby = paba - f
        s = np.where(np.logical_and(cbx < 0.0, cay < 0.0), -1.0, 1.0)
        return s * np.sqrt(
            np.minimum(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba)
        )

    def get_bounding_box(self) -> AABB | None:
        a, b, ra, rb = self.a, self.b, self.ra, self.rb
        up = a - b
        vx = project_vector_on_plane((1, 0, 0), up)
        vy = project_vector_on_plane((0, 1, 0), up)
        vz = project_vector_on_plane((0, 0, 1), up)
        a1 = a - ra * vx - ra * vy - ra * vz
        a2 = a + ra * vx + ra * vy + ra * vz
        b1 = b - rb * vx - rb * vy - rb * vz
        b2 = b + rb * vx + rb * vy + rb * vz
        return (
            np.minimum(a1, b1).astype(np.float32),
            np.maximum(a2, b2).astype(np.float32),
        )


class SDFRoundCone(SDF):
    """Round cone is made up of two balls and a cylinder."""

    def __init__(
        self, a: npt.ArrayLike, b: npt.ArrayLike, ra: float, rb: float
    ) -> None:
        """SDF of round cone.

        Parmeters
        ---------
        a, b : ArrayLike
            Coordinates of point A/B of shape (3,).
        ra, rb : float
            Radius of point A/B.
        """

        self.a = np.array(a, dtype=np.float32)
        self.b = np.array(b, dtype=np.float32)
        self.ra = ra
        self.rb = rb

        assert tuple(self.a.shape) == (3,), "a should be vector of 3d"
        assert tuple(self.b.shape) == (3,), "b should be vector of 3d"

        self.bounding_box = (
            np.min([self.a - self.ra, self.b - self.rb], axis=0).astype(np.float32),
            np.max([self.a + self.ra, self.b + self.rb], axis=0).astype(np.float32),
        )

    def distance(self, p: npt.ArrayLike) -> npt.NDArray[np.float32]:
        # pylint: disable=too-many-locals
        p = np.array(p, dtype=np.float32)
        assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

        a = self.a
        b = self.b
        ra = self.ra
        rb = self.rb

        # sampling independent computations (only depend on shape)
        ba = b - a
        l2 = np.dot(ba, ba)
        rr = ra - rb
        a2 = l2 - rr * rr
        il2 = 1.0 / l2

        # sampling dependant computations
        pa = p - a
        y = np.dot(pa, ba)
        z = y - l2
        x = pa * l2 - np.outer(y, ba)
        x2 = np.sum(x * x, axis=1)
        y2 = y * y * l2
        z2 = z * z * l2

        # single square root!
        k = np.sign(rr) * rr * rr * x2
        dis = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - ra

        lt = np.sign(z) * a2 * z2 > k
        dis[lt] = np.sqrt(x2[lt] + z2[lt]) * il2 - rb

        rt = np.sign(y) * a2 * y2 < k
        dis[rt] = np.sqrt(x2[rt] + y2[rt]) * il2 - ra

        return dis
