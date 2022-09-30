"""Signed distance functions.

Refs: https://iquilezles.org/articles/distfunctions/
"""

from typing import Callable, List

import numpy as np
import numpy.typing as npt

__all__ = ["SDF", "SDFs", "SDFIsIn", "sdf_compose", "sdf_is_in", "sdf_round_cone"]

SDF = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
SDFs = List[SDF]
SDFIsIn = Callable[[npt.NDArray[np.float32]], npt.NDArray[np.bool_]]


def sdf_compose(sdfs: SDFs) -> SDF:
    def compose(p: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.min([sdf(p) for sdf in sdfs], axis=0)

    return compose


def sdf_is_in(sdfs: SDFs) -> SDFIsIn:
    compose = sdf_compose(sdfs)  # TODO: perf

    def is_in(p: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
        return compose(p) <= 0

    return is_in


# pylint: disable=too-many-locals
def sdf_round_cone(
    p: npt.ArrayLike, a: npt.ArrayLike, b: npt.ArrayLike, ra: float, rb: float
) -> npt.NDArray[np.float32]:
    """SDF of round cone.

    Parmeters
    ---------
    p: ArrayLike
        Hit point p of shape (N, 3).
    a: ArrayLike
        Coordinates of point A of shape (3,).
    b: ArrayLike
        Coordinates of point A of shape (3,).
    ra: float
        Radius of point A.
    rb: float
        Radius of point B.

    Returns
    -------
    distance : npt.NDArray[np.float32]
        Distance array of shape (3,).
    """

    p = np.array(p, dtype=np.float32)
    assert p.ndim == 2 and p.shape[1] == 3, "p should be array of shape (N, 3)"

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    assert tuple(a.shape) == (3,), "a should be vector of 3d"
    assert tuple(b.shape) == (3,), "b should be vector of 3d"

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
