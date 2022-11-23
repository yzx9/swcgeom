"""Signed distance functions.

Refs: https://iquilezles.org/articles/distfunctions/
"""

import warnings
from typing import List, Tuple, Iterable

import numpy as np
import numpy.typing as npt

__all__ = ["SDF", "SDFCompose", "SDFRoundCone"]

# Axis-aligned bounding box, tuple of array of shape (3,)
AABB = Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]


class SDF:
    """Signed distance functions."""

    bounding_box: AABB | None = None

    def __call__(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return self.distance(p)

    def distance(self, p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Calc signed distance.

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


class SDFCompose(SDF):
    """Compose multiple SDFs."""

    def __init__(self, sdfs: List[SDF]) -> None:
        assert len(sdfs) != 0, "must combine at least one SDF"

        if len(sdfs) == 1:
            warnings.warn("compose only one SDF, use SDFCompose.compose instead")

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
        in_box = self.is_in_bounding_box(p)
        is_in = np.stack([sdf.is_in(p[in_box]) for sdf in self.sdfs])
        flags = np.full_like(in_box, False, dtype=np.bool_)
        flags[in_box] = np.any(is_in, axis=0)
        return flags

    @staticmethod
    def compose(sdfs: Iterable[SDF]) -> SDF:
        sdfs = list(sdfs)
        return SDFCompose(sdfs) if len(sdfs) != 1 else sdfs[0]


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
