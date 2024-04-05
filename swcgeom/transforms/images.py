"""Image stack related transform."""

import warnings

import numpy as np
import numpy.typing as npt

from swcgeom.transforms.base import Identity, Transform

__all__ = [
    "ImagesCenterCrop",
    "ImagesScale",
    "ImagesClip",
    "ImagesFlip",
    "ImagesFlipY",
    "ImagesNormalizer",
    "ImagesMeanVarianceAdjustment",
    "ImagesScaleToUnitRange",
    "ImagesHistogramEqualization",
    "Center",  # legacy
]


NDArrayf32 = npt.NDArray[np.float32]


class ImagesCenterCrop(Transform[NDArrayf32, NDArrayf32]):
    """Get image stack center."""

    def __init__(self, shape_out: int | tuple[int, int, int]):
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

    .. deprecated:: 0.16.0
        Use :class:`ImagesCenterCrop` instead.
    """

    def __init__(self, shape_out: int | tuple[int, int, int]):
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

    def extra_repr(self) -> str:
        return f"scaler={self.scaler}"


class ImagesClip(Transform[NDArrayf32, NDArrayf32]):
    def __init__(self, vmin: float = 0, vmax: float = 1, /) -> None:
        super().__init__()
        self.vmin, self.vmax = vmin, vmax

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        return np.clip(x, self.vmin, self.vmax)

    def extra_repr(self) -> str:
        return f"vmin={self.vmin}, vmax={self.vmax}"


class ImagesFlip(Transform[NDArrayf32, NDArrayf32]):
    """Flip image stack along axis."""

    def __init__(self, axis: int, /) -> None:
        super().__init__()
        self.axis = axis

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        return np.flip(x, axis=self.axis)

    def extra_repr(self) -> str:
        return f"axis={self.axis}"


class ImagesFlipY(ImagesFlip):
    """Flip image stack along Y-axis.

    See Also
    --------
    ~.images.io.TeraflyImageStack:
        Terafly and Vaa3d use a especial right-handed coordinate system
        (with origin point in the left-top and z-axis points front),
        but we flip y-axis to makes it a left-handed coordinate system
        (with orgin point in the left-bottom and z-axis points front).
        If you need to use its coordinate system, remember to FLIP
        Y-AXIS BACK.
    """

    def __init__(self, axis: int = 1, /) -> None:
        super().__init__(axis)  # (X, Y, Z, C)


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

    def extra_repr(self) -> str:
        return f"mean={self.mean}, variance={self.variance}"


class ImagesScaleToUnitRange(Transform[NDArrayf32, NDArrayf32]):
    """Scale image stack to unit range."""

    def __init__(self, vmin: float, vmax: float, *, clip: bool = True) -> None:
        """Scale image stack to unit range.

        Parameters
        ----------
        vmin : float
            Minimum value.
        vmax : float
            Maximum value.
        clip : bool, default True
            Clip values to [0, 1] to avoid numerical issues.
        """

        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.diff = vmax - vmin
        self.clip = clip
        self.post = ImagesClip(0, 1) if self.clip else Identity()

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        return self.post((x - self.vmin) / self.diff)

    def extra_repr(self) -> str:
        return f"vmin={self.vmin}, vmax={self.vmax}, clip={self.clip}"


class ImagesHistogramEqualization(Transform[NDArrayf32, NDArrayf32]):
    """Image histogram equalization.

    References
    ----------
    http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    """

    def __init__(self, bins: int = 256) -> None:
        super().__init__()
        self.bins = bins

    def __call__(self, x: NDArrayf32) -> NDArrayf32:
        # get image histogram
        hist, bin_edges = np.histogram(x.flatten(), self.bins, density=True)
        cdf = hist.cumsum()  # cumulative distribution function
        cdf = cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        equalized = np.interp(x.flatten(), bin_edges[:-1], cdf)
        return equalized.reshape(x.shape).astype(np.float32)

    def extra_repr(self) -> str:
        return f"bins={self.bins}"
