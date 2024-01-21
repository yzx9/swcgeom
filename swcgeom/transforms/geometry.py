"""SWC geometry operations."""

import warnings
from typing import Generic, Literal, Optional, TypeVar

import numpy as np
import numpy.typing as npt

from swcgeom.core import DictSWC
from swcgeom.core.swc_utils import SWCNames
from swcgeom.transforms.base import Transform
from swcgeom.utils import (
    rotate3d,
    rotate3d_x,
    rotate3d_y,
    rotate3d_z,
    scale3d,
    translate3d,
)

__all__ = [
    "Normalizer",
    "RadiusReseter",
    "AffineTransform",
    "Translate",
    "TranslateOrigin",
    "Scale",
    "Rotate",
    "RotateX",
    "RotateY",
    "RotateZ",
]

T = TypeVar("T", bound=DictSWC)
Center = Literal["root", "soma", "origin"]


# pylint: disable=too-few-public-methods
class Normalizer(Generic[T], Transform[T, T]):
    """Noramlize coordinates and radius to 0-1."""

    def __init__(self, *, names: Optional[SWCNames] = None) -> None:
        super().__init__()
        if names is not None:
            warnings.warn(
                "`name` parameter is no longer needed, now use the "
                "built-in names table, you can directly remove it.",
                DeprecationWarning,
            )

    def __call__(self, x: T) -> T:
        """Scale the `x`, `y`, `z`, `r` of nodes to 0-1."""
        new_tree = x.copy()
        xyzr = [x.names.x, x.names.y, x.names.z, x.names.r]
        for key in xyzr:  # TODO: does r is the same?
            vs = new_tree.ndata[key]
            new_tree.ndata[key] = (vs - np.min(vs)) / np.max(vs)

        return new_tree


class RadiusReseter(Generic[T], Transform[T, T]):
    """Reset radius to fixed value."""

    def __init__(self, r: float) -> None:
        super().__init__()
        self.r = r

    def __call__(self, x: T) -> T:
        r = np.full_like(x.r(), fill_value=self.r)
        new_tree = x.copy()
        new_tree.ndata[new_tree.names.r] = r
        return new_tree

    def extra_repr(self):
        return f"r={self.r:.4f}"


class AffineTransform(Generic[T], Transform[T, T]):
    """Apply affine matrix."""

    tm: npt.NDArray[np.float32]
    center: Center
    fmt: str

    def __init__(
        self,
        tm: npt.NDArray[np.float32],
        center: Center = "origin",
        *,
        fmt: Optional[str] = None,
        names: Optional[SWCNames] = None,
    ) -> None:
        self.tm, self.center = tm, center

        if fmt is not None:
            warnings.warn(
                "`fmt` parameter is no longer needed, now use the "
                "extra_repr(), you can directly remove it.",
                DeprecationWarning,
            )

        if names is not None:
            warnings.warn(
                "`name` parameter is no longer needed, now use the "
                "built-in names table, you can directly remove it.",
                DeprecationWarning,
            )

    def __call__(self, x: T) -> T:
        match self.center:
            case "root" | "soma":
                idx = np.nonzero(x.ndata[x.names.pid] == -1)[0][0].item()
                xyz = x.xyz()[idx]
                tm = (
                    translate3d(-xyz[0], -xyz[1], -xyz[2])
                    .dot(self.tm)
                    .dot(translate3d(xyz[0], xyz[1], xyz[2]))
                )
            case _:
                tm = self.tm

        return self.apply(x, tm)

    @staticmethod
    def apply(x: T, tm: npt.NDArray[np.float32]) -> T:
        xyzw = x.xyzw().dot(tm.T).T
        xyzw /= xyzw[3]

        y = x.copy()
        y.ndata[x.names.x] = xyzw[0]
        y.ndata[x.names.y] = xyzw[1]
        y.ndata[x.names.z] = xyzw[2]
        return y


class Translate(Generic[T], AffineTransform[T]):
    """Translate SWC."""

    def __init__(self, tx: float, ty: float, tz: float, **kwargs) -> None:
        super().__init__(translate3d(tx, ty, tz), **kwargs)
        self.tx, self.ty, self.tz = tx, ty, tz

    def extra_repr(self):
        return f"tx={self.tx:.4f}, ty={self.ty:.4f}, tz={self.tz:.4f}"

    @classmethod
    def transform(cls, x: T, tx: float, ty: float, tz: float, **kwargs) -> T:
        return cls(tx, ty, tz, **kwargs)(x)


class TranslateOrigin(Generic[T], Transform[T, T]):
    """Translate root of SWC to origin point."""

    def __call__(self, x: T) -> T:
        return self.transform(x)

    @classmethod
    def transform(cls, x: T) -> T:
        pid = np.nonzero(x.ndata[x.names.pid] == -1)[0][0].item()
        xyzw = x.xyzw()
        tm = translate3d(-xyzw[pid, 0], -xyzw[pid, 1], -xyzw[pid, 2])
        return AffineTransform.apply(x, tm)


class Scale(Generic[T], AffineTransform[T]):
    """Scale SWC."""

    def __init__(
        self, sx: float, sy: float, sz: float, center: Center = "root", **kwargs
    ) -> None:
        super().__init__(scale3d(sx, sy, sz), center=center, **kwargs)

    @classmethod
    def transform(  # pylint: disable=too-many-arguments
        cls, x: T, sx: float, sy: float, sz: float, center: Center = "root", **kwargs
    ) -> T:
        return cls(sx, sy, sz, center=center, **kwargs)(x)


class Rotate(Generic[T], AffineTransform[T]):
    """Rotate SWC."""

    def __init__(
        self,
        n: npt.NDArray[np.float32],
        theta: float,
        center: Center = "root",
        **kwargs,
    ) -> None:
        fmt = f"Rotate-{n[0]}-{n[1]}-{n[2]}-{theta:.4f}"
        super().__init__(rotate3d(n, theta), center=center, fmt=fmt, **kwargs)
        self.n = n
        self.theta = theta
        self.center = center

    def extra_repr(self):
        return f"n={self.n}, theta={self.theta:.4f}, center={self.center}"  # TODO: imporve format of n

    @classmethod
    def transform(
        cls,
        x: T,
        n: npt.NDArray[np.float32],
        theta: float,
        center: Center = "root",
        **kwargs,
    ) -> T:
        return cls(n, theta, center=center, **kwargs)(x)


class RotateX(Generic[T], AffineTransform[T]):
    """Rotate SWC with x-axis."""

    def __init__(self, theta: float, center: Center = "root", **kwargs) -> None:
        super().__init__(rotate3d_x(theta), center=center, **kwargs)
        self.theta = theta

    def extra_repr(self):
        return f"center={self.center}, theta={self.theta:.4f}"

    @classmethod
    def transform(cls, x: T, theta: float, center: Center = "root", **kwargs) -> T:
        return cls(theta, center=center, **kwargs)(x)


class RotateY(Generic[T], AffineTransform[T]):
    """Rotate SWC with y-axis."""

    def __init__(self, theta: float, center: Center = "root", **kwargs) -> None:
        super().__init__(rotate3d_y(theta), center=center, **kwargs)
        self.theta = theta
        self.center = center

    def extra_repr(self):
        return f"theta={self.theta:.4f}, center={self.center}"

    @classmethod
    def transform(cls, x: T, theta: float, center: Center = "root", **kwargs) -> T:
        return cls(theta, center=center, **kwargs)(x)


class RotateZ(Generic[T], AffineTransform[T]):
    """Rotate SWC with z-axis."""

    def __init__(self, theta: float, center: Center = "root", **kwargs) -> None:
        super().__init__(rotate3d_z(theta), center=center, **kwargs)
        self.theta = theta
        self.center = center

    def extra_repr(self):
        return f"theta={self.theta:.4f}, center={self.center}"

    @classmethod
    def transform(cls, x: T, theta: float, center: Center = "root", **kwargs) -> T:
        return cls(theta, center=center, **kwargs)(x)
