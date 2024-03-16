"""Image stack related transform."""

import warnings
from typing import Tuple

import numpy as np
import numpy.typing as npt

from swcgeom.transforms.base import Transform

__all__ = [
    "ImagesCenterCrop",
    "ImagesScale",
    "ImagesClip",
    "ImagesNormalizer",
    "ImagesMeanVarianceAdjustment",
    "Center",  # legacy
]


NDArrayf32 = npt.NDArray[np.float32]


class ImagesCenterCrop(Transform[NDArrayf32, NDArrayf32]):
    """Get image stack center."""

    def __init__(self, shape_out: int | Tuple[int, int, int]):
        super().__init__()
        self.shape_out = (
            shape_out
            if isinstance(shape_out, tuple)
            else (shape_out, shape_out, shape_out)
        )

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        diff = np.subtract(x.shape[:3], self.shape_out)
        s = diff // 2
        e = np.add(s, self.shape_out)
        return x[s[0] : e[0], s[1] : e[1], s[2] : e[2], :]

    def extra_repr(self) -> str:
        return f"shape_out=({','.join(str(a) for a in self.shape_out)})"


class Center(ImagesCenterCrop):
    """Get image stack center.

    .. deprecated:: 0.5.0
        Use :class:`ImagesCenterCrop` instead.
    """

    def __init__(self, shape_out: int | Tuple[int, int, int]):
        warnings.warn(
            "`Center` is deprecated, use `ImagesCenterCrop` instead",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(shape_out)


class ImagesScale(Transform[NDArrayf32, NDArrayf32]):
    def __init__(self, scaler: float) -> None:
        super().__init__()
        self.scaler = scaler

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        return self.scaler * x


class ImagesClip(Transform[NDArrayf32, NDArrayf32]):
    def __init__(self, vmin: float = 0, vmax: float = 1, /) -> None:
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        return np.clip(x, self.vmin, self.vmax)


class ImagesNormalizer(Transform[NDArrayf32, NDArrayf32]):
    """Normalize image stack."""

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        mean = np.mean(x)
        variance = np.var(x)
        return (x - mean) / variance


class ImagesMeanVarianceAdjustment(Transform[NDArrayf32, NDArrayf32]):
    """Adjust image stack mean and variance.

    See Also
    --------
    ~swcgeom.images.ImageStackFolder.stat
    """

    def __init__(self, mean: float, variance: float) -> None:
        super().__init__()
        self.mean = mean
        self.variance = variance

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        return (x - self.mean) / self.variance
