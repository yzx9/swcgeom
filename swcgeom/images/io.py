
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Read and write image stack."""

import os
import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from functools import cache, lru_cache
from typing import Any, Generic, Literal, TypeVar, cast, overload

import nrrd
import numpy as np
import numpy.typing as npt
import tifffile
from typing_extensions import deprecated

from swcgeom.images.loaders import PBD, Raw

__all__ = ["read_imgs", "save_tiff", "read_images"]

Vec3i = tuple[int, int, int]
ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)

RE_TERAFLY_ROOT = re.compile(r"^RES\((\d+)x(\d+)x(\d+)\)$")
RE_TERAFLY_NAME = re.compile(r"^\d+(_\d+)?(_\d+)?")

UINT_MAX = {
    np.dtype(np.uint8): (2**8) - 1,
    np.dtype(np.uint16): (2**16) - 1,
    np.dtype(np.uint32): (2**32) - 1,
    np.dtype(np.uint64): (2**64) - 1,
}

AXES_ORDER = {
    "X": 0,
    "Y": 1,
    "Z": 2,
    "C": 3,
    "I": 2,  # vaa3d compatibility
}


class ImageStack(ABC, Generic[ScalarType]):
    """Image stack."""

    # fmt: off
    @overload
    @abstractmethod
    def __getitem__(self, key: int) -> npt.NDArray[ScalarType]: ...                     # array of shape (Y, Z, C)
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int]) -> npt.NDArray[ScalarType]: ...         # array of shape (Z, C)
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int, int]) -> npt.NDArray[ScalarType]: ...    # array of shape (C,)
    @overload
    @abstractmethod
    def __getitem__(self, key: tuple[int, int, int, int]) -> ScalarType: ...            # value
    @overload
    @abstractmethod
    def __getitem__(
        self, key: slice | tuple[slice, slice] | tuple[slice, slice, slice] |  tuple[slice, slice, slice, slice],
    ) -> npt.NDArray[ScalarType]: ... # array of shape (X, Y, Z, C)
    @overload
    @abstractmethod
    def __getitem__(self, key: npt.NDArray[np.integer[Any]]) -> npt.NDArray[ScalarType]: ...
    # fmt: on
    @abstractmethod
    def __getitem__(self, key):
        """Get pixel/patch of image stack.

        Returns:
            value: NDArray which shape depends on key. If key is tuple of ints,
        """
        raise NotImplementedError()

    def get_full(self) -> npt.NDArray[ScalarType]:
        """Get full image stack.

        NOTE: this will load the full image stack into memory.
        """
        return self[:, :, :, :]

    @property
    def shape(self) -> tuple[int, int, int, int]:
        raise NotImplementedError()


@overload
def read_imgs(fname: str, *, dtype: ScalarType, **kwargs) -> ImageStack[ScalarType]: ...
@overload
def read_imgs(fname: str, *, dtype: None = ..., **kwargs) -> ImageStack[np.float32]: ...
def read_imgs(fname: str, **kwargs):  # type: ignore
    """Read image stack.

    Args:
        fname: The path of image stack.
        dtype: Casting data to specified dtype.
            If integer and float conversions occur, they will be scaled (assuming floats
            are between 0 and 1). Default to `np.float32`.
        **kwargs: Forwarding to the corresponding reader.
    """
    kwargs.setdefault("dtype", np.float32)
    if not os.path.exists(fname):
        raise ValueError(f"image stack not exists: {fname}")

    # match file extension
    match os.path.splitext(fname)[-1]:
        case ".tif" | ".tiff":
            return TiffImageStack(fname, **kwargs)
        case ".nrrd":
            return NrrdImageStack(fname, **kwargs)
        case ".v3dpbd":
            return V3dpbdImageStack(fname, **kwargs)
        case ".v3draw":
            return V3drawImageStack(fname, **kwargs)
        case ".npy":
            return NDArrayImageStack(np.load(fname), **kwargs)

    # try to read as terafly
    if TeraflyImageStack.is_root(fname):
        return TeraflyImageStack(fname, **kwargs)

    raise ValueError("unsupported image stack")


def save_tiff(
    data: npt.NDArray | ImageStack,
    fname: str,
    *,
    dtype: np.unsignedinteger | np.floating | None = None,
    compression: str | Literal[False] = "zlib",
    **kwargs,
) -> None:
    """Save image stack as tiff.

    Args:
        data: The image stack.
        fname: str
        dtype: Casting data to specified dtype.
            If integer and float conversions occur, they will be scaled (assuming
            floats are between 0 and 1).
        compression: Compression algorithm, forwarding to `tifffile.imwrite`.
            If no algorithnm is specify specified, we will use the zlib algorithm with
            compression level 6 by default.
        **kwargs: Forwarding to `tifffile.imwrite`
    """
    if isinstance(data, ImageStack):
        data = data.get_full()  # TODO: avoid load full imgs to memory

    if data.ndim == 3:
        data = np.expand_dims(data, -1)  # (_, _, _)  -> (_, _, _, C), C === 1

    axes = "ZXYC"
    assert data.ndim == 4, "should be an array of shape (X, Y, Z, C)"
    assert data.shape[-1] in [1, 3], "support 'miniblack' or 'rgb'"

    if dtype is not None:
        if np.issubdtype(data.dtype, np.floating) and np.issubdtype(
            dtype, np.unsignedinteger
        ):
            scaler_factor = UINT_MAX[np.dtype(dtype)]
        elif np.issubdtype(data.dtype, np.unsignedinteger) and np.issubdtype(
            dtype, np.floating
        ):
            scaler_factor = 1 / UINT_MAX[np.dtype(data.dtype)]
        else:
            scaler_factor = 1

        data = (data * scaler_factor).astype(dtype)

    if compression is not False:
        kwargs.setdefault("compression", compression)
        if compression == "zlib":
            kwargs.setdefault("compressionargs", {"level": 6})

    data = np.moveaxis(data, 2, 0)  # (_, _, Z, _) -> (Z, _, _, _)
    kwargs.setdefault("photometric", "rgb" if data.shape[-1] == 3 else "minisblack")
    metadata = kwargs.get("metadata", {})
    metadata.setdefault("axes", axes)
    kwargs.update(metadata=metadata)
    tifffile.imwrite(fname, data, **kwargs)


class NDArrayImageStack(ImageStack[ScalarType]):
    """NDArray image stack."""

    def __init__(
        self, imgs: npt.NDArray[Any], *, dtype: ScalarType | None = None
    ) -> None:
        super().__init__()

        if imgs.ndim == 3:  # (_, _, _) -> (_, _, _, C)
            imgs = np.expand_dims(imgs, -1)
        assert imgs.ndim == 4, "Should be shape of (X, Y, Z, C)"

        if dtype is not None:
            dtype_raw = imgs.dtype
            if np.issubdtype(dtype, np.floating) and np.issubdtype(
                dtype_raw, np.unsignedinteger
            ):
                scalar_factor = 1.0 / UINT_MAX[dtype_raw]
                imgs = scalar_factor * imgs.astype(dtype)
            elif np.issubdtype(dtype, np.unsignedinteger) and np.issubdtype(
                dtype_raw, np.floating
            ):
                scalar_factor = UINT_MAX[dtype]
                imgs *= (scalar_factor * imgs).astype(dtype)
            else:
                imgs = imgs.astype(dtype)

        self.imgs = imgs

    def __getitem__(self, key):
        return self.imgs.__getitem__(key)

    def get_full(self) -> npt.NDArray[ScalarType]:
        return self.imgs

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return cast(tuple[int, int, int, int], self.imgs.shape)


class TiffImageStack(NDArrayImageStack[ScalarType]):
    """Tiff image stack."""

    def __init__(self, fname: str, *, dtype: ScalarType, **kwargs) -> None:
        with tifffile.TiffFile(fname, **kwargs) as f:
            s = f.series[0]
            imgs, axes = s.asarray(), s.axes

        if len(axes) != imgs.ndim or any(c not in AXES_ORDER for c in axes):
            axes_raw = axes
            axes = "ZXYC" if imgs.ndim == 4 else "ZXY"
            warnings.warn(f"reset unexcept axes `{axes_raw}` to `{axes}` in: {fname}")

        orders = [AXES_ORDER[c] for c in axes]
        imgs = imgs.transpose(np.argsort(orders))
        super().__init__(imgs, dtype=dtype)


class NrrdImageStack(NDArrayImageStack[ScalarType]):
    """Nrrd image stack."""

    def __init__(self, fname: str, *, dtype: ScalarType, **kwargs) -> None:
        imgs, header = nrrd.read(fname, **kwargs)
        super().__init__(imgs, dtype=dtype)
        self.header = header


class V3dImageStack(NDArrayImageStack[ScalarType]):
    """v3d image stack."""

    def __init_subclass__(cls, loader: Raw | PBD) -> None:
        super().__init_subclass__()
        cls._loader = loader

    def __init__(self, fname: str, *, dtype: ScalarType, **kwargs) -> None:
        r = self._loader()
        imgs = r.load(fname)
        super().__init__(imgs, dtype=dtype, **kwargs)


class V3drawImageStack(V3dImageStack[ScalarType], loader=Raw):
    """v3draw image stack."""


class V3dpbdImageStack(V3dImageStack[ScalarType], loader=PBD):
    """v3dpbd image stack."""


class TeraflyImageStack(ImageStack[ScalarType]):
    """TeraFly image stack.

    TeraFly is a terabytes of multidimensional volumetric images file
    format as described in [1]_.

    NOTE: Terafly and Vaa3d use a especial right-handed coordinate system
    (with origin point in the left-top and z-axis points front), but we
    flip y-axis to makes it a left-handed coordinate system (with origin
    point in the left-bottom and z-axis points front). If you need to
    use its coordinate system, remember to FLIP Y-AXIS BACK.

    References:
    .. [1] Bria, Alessandro, Giulio Iannello, Leonardo Onofri, and Hanchuan Peng.
       “TeraFly: Real-Time Three-Dimensional Visualization and Annotation of Terabytes
       of Multidimensional Volumetric Images.” Nature Methods 13, no. 3 (March 2016):
       192-94. https://doi.org/10.1038/nmeth.3767.
    """

    _listdir: Callable[[str], list[str]]
    _read_patch: Callable[[str], npt.NDArray]

    def __init__(
        self, root: str, *, dtype: ScalarType, lru_maxsize: int | None = 128
    ) -> None:
        r"""
        Args:
            root: The root of terafly which contains directories named as `RES(YxXxZ)`.
            dtype: np.dtype
            lru_maxsize: Forwarding to `functools.lru_cache`.
                A decompressed array size of (256, 256, 256, 1), which is the typical
                size of terafly image stack, takes about 256 * 256 * 256 * 1 * 4B = 64MB.
                A cache size of 128 requires about 8GB memory.
        """

        super().__init__()
        self.root = root
        self.dtype = dtype
        self.res, self.res_dirs, self.res_patch_sizes = self.get_resolutions(root)

        @cache
        def listdir(path: str) -> list[str]:
            return os.listdir(path)

        @lru_cache(maxsize=lru_maxsize)
        def read_patch(path: str) -> npt.NDArray[ScalarType]:
            match os.path.splitext(path)[-1]:
                case "raw":
                    # Treat it as a v3draw file
                    return V3drawImageStack(path, dtype=dtype).get_full()
                case _:
                    return read_imgs(path, dtype=dtype).get_full()

        self._listdir, self._read_patch = listdir, read_patch

    def __getitem__(self, key):
        """Get images in max resolution.

        >>> imgs[0, 0, 0, 0]  # get value # doctest: +SKIP
        >>> imgs[0:64, 0:64, 0:64, :]  # get patch # doctest: +SKIP
        ```
        """
        if not isinstance(key, tuple):
            raise IndexError(
                "Potential memory issue, you are loading large images "
                "into memory, if sure, load it explicitly with "
                "`get_full`"
            )

        if not isinstance(key[0], slice):
            offset = [key[i] for i in range(3)]
            return self.get_patch(offset, np.add(offset, 1)).item()

        slices = [k.indices(self.res[-1][i]) for i, k in enumerate(key)]
        starts, ends, strides = np.array(slices).transpose()
        return self.get_patch(starts, ends, strides)

    def get_patch(
        self, starts, ends, strides: int | Vec3i = 1, res_level=-1
    ) -> npt.NDArray[ScalarType]:
        """Get patch of image stack.

        Returns:
            patch: array of shape (X, Y, Z, C)
        """
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        starts, ends = np.array(starts), np.array(ends)
        self._check_params(res_level, starts, np.subtract(ends, 1))
        assert np.equal(strides, [1, 1, 1]).all()  #  TODO: support stride

        shape_out = np.concatenate([ends - starts, [1]])
        out = np.zeros(shape_out, dtype=self.dtype)
        self._get_range(starts, ends, res_level, out=out)

        # flip y-axis to makes it a left-handed coordinate system
        out = np.flip(out, axis=1)
        return out

    def find_correspond_imgs(self, p, res_level=-1):
        """Find the image which contain this point.

        Returns:
            patch: array of shape (X, Y, Z, C)
            patch_offset: (int, int, int)
        """
        p = np.array(p)
        self._check_params(res_level, p)
        return self._find_correspond_imgs(p, res_level)

    def get_correspond_coord(self, p, in_res_level: int, out_res_level: int):
        raise NotImplementedError()  # TODO

    @property
    def shape(self) -> tuple[int, int, int, int]:
        res_max = self.res[-1]
        return res_max[0], res_max[1], res_max[2], 1

    @classmethod
    def get_resolutions(cls, root: str) -> tuple[list[Vec3i], list[str], list[Vec3i]]:
        """Get all resolutions.

        Returns:
            resolutions: Sequence of sorted resolutions (from small to large).
            roots: Sequence of root of resolutions respectively.
            patch_sizes: Sequence of patch size of resolutions respectively.
        """

        roots = list(cls.get_resolution_dirs(root))
        assert len(roots) > 0, "no resolution detected"

        res = [RE_TERAFLY_ROOT.search(d) for d in roots]
        res = [[int(a) for a in d.groups()] for d in res if d is not None]
        res = np.array(res)
        res[:, [0, 1]] = res[:, [1, 0]]  # (Y, X, _) -> (X, Y, _)

        def listdir(d: str):
            return filter(RE_TERAFLY_NAME.match, os.listdir(d))

        def get_patch_size(src: str):
            y0 = next(listdir(src))
            x0 = next(listdir(os.path.join(src, y0)))
            z0 = next(listdir(os.path.join(src, y0, x0)))
            patch = read_imgs(os.path.join(src, y0, x0, z0))
            return patch.shape

        patch_sizes = [get_patch_size(os.path.join(root, d)) for d in roots]

        # sort
        indices = np.argsort(np.prod(res, axis=1, dtype=np.longlong))
        res = res[indices]
        roots = np.take(roots, indices)
        patch_sizes = np.take(patch_sizes, indices)
        return res, roots, patch_sizes  # type: ignore

    @staticmethod
    def is_root(root: str) -> bool:
        return os.path.isdir(root) and any(
            RE_TERAFLY_ROOT.match(d) for d in os.listdir(root)
        )

    @staticmethod
    def get_resolution_dirs(root: str) -> Iterable[str]:
        return filter(RE_TERAFLY_ROOT.match, os.listdir(root))

    def _check_params(self, res_level, *coords):
        assert res_level == -1  # TODO: support multi-resolutions

        res_level = len(self.res) + res_level if res_level < 0 else res_level
        assert 0 <= res_level < len(self.res), "invalid resolution level"

        res = self.res[res_level]
        for p in coords:
            assert np.less([0, 0, 0], p).all(), (
                f"indices ({p[0]}, {p[1]}, {p[2]}) out of range (0, 0, 0)"
            )

            assert np.greater(res, p).all(), (
                f"indices ({p[0]}, {p[1]}, {p[2]}) out of range ({res[0]}, {res[1]}, {res[2]})"
            )

    def _get_range(self, starts, ends, res_level, out):
        # pylint: disable=too-many-locals
        shape = ends - starts
        patch, offset = self._find_correspond_imgs(starts, res_level=res_level)
        if patch is not None:
            coords = starts - offset
            lens = np.min([patch.shape[:3] - coords, shape], axis=0)
            out[: lens[0], : lens[1], : lens[2]] = patch[
                coords[0] : coords[0] + lens[0],
                coords[1] : coords[1] + lens[1],
                coords[2] : coords[2] + lens[2],
            ]
        else:
            size = self.res_patch_sizes[res_level]
            lens = (np.floor(starts / size).astype(np.int64) + 1) * size - starts

        if shape[0] > lens[0]:
            starts_x = starts + [lens[0], 0, 0]
            ends_x = ends
            self._get_range(starts_x, ends_x, res_level, out[lens[0] :, :, :])

        if shape[1] > lens[1]:
            starts_y = starts + [0, lens[1], 0]
            ends_y = np.array([starts[0], ends[1], ends[2]])
            ends_y += [min(shape[0], lens[0]), 0, 0]
            self._get_range(starts_y, ends_y, res_level, out[:, lens[1] :, :])

        if shape[2] > lens[2]:
            starts_z = starts + [0, 0, lens[2]]
            ends_z = np.array([starts[0], starts[1], ends[2]])
            ends_z += [min(shape[0], lens[0]), min(shape[1], lens[1]), 0]
            self._get_range(starts_z, ends_z, res_level, out[:, :, lens[2] :])

    def _find_correspond_imgs(self, p, res_level):
        # pylint: disable=too-many-locals
        x, y, z = p
        cur = os.path.join(self.root, self.res_dirs[res_level])

        def get_v(f: str):
            return float(os.path.splitext(f.split("_")[-1])[0])

        for v in [y, x, z]:
            # extract v from `y/`, `y_x/`, `y_x_z.tif`
            dirs = [d for d in self._listdir(cur) if RE_TERAFLY_NAME.match(d)]
            diff = np.array([get_v(d) for d in dirs])
            if (invalid := diff > 10 * v).all():
                return None, None

            diff[invalid] = -np.inf  # remove values which greater than v

            # find the index of the value smaller than v and closest to v
            idx = np.argmax(diff)
            cur = os.path.join(cur, dirs[idx])

        patch = self._read_patch(cur)
        name = os.path.splitext(os.path.basename(cur))[0]
        offset = [int(int(i) / 10) for i in name.split("_")]
        offset[0], offset[1] = offset[1], offset[0]  # (Y, X, _) -> (X, Y, _)
        if np.less_equal(np.add(offset, patch.shape[:3]), p).any():
            return None, None

        return patch, offset


# Legacy


class GrayImageStack:
    """Gray Image stack."""

    imgs: ImageStack

    def __init__(self, imgs: ImageStack) -> None:
        self.imgs = imgs

    @overload
    def __getitem__(self, key: Vec3i) -> np.float32: ...
    @overload
    def __getitem__(self, key: npt.NDArray[np.integer[Any]]) -> np.float32: ...
    @overload
    def __getitem__(
        self, key: slice | tuple[slice, slice] | tuple[slice, slice, slice]
    ) -> npt.NDArray[np.float32]: ...
    def __getitem__(self, key):
        """Get pixel/patch of image stack."""
        v = self[key]
        if not isinstance(v, np.ndarray):
            return v
        if v.ndim == 4:
            return v[:, :, :, 0]
        if v.ndim == 3:
            return v[:, :, 0]
        if v.ndim == 2:
            return v[:, 0]
        if v.ndim == 1:
            return v[0]
        raise ValueError("unsupported key")

    def get_full(self) -> npt.NDArray[np.float32]:
        """Get full image stack.

        NOTE: this will load the full image stack into memory.
        """
        return self.imgs.get_full()[:, :, :, 0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.imgs.shape[:-1]


@deprecated("Use `read_imgs` instead")
def read_images(*args, **kwargs) -> GrayImageStack:
    """Read images.

    .. deprecated:: 0.16.0
        Use :meth:`read_imgs` instead.
    """

    return GrayImageStack(read_imgs(*args, **kwargs))
