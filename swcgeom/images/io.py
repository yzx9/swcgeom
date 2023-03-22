"""Read and write image stack."""


import os
import re
from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Tuple, overload

import nrrd
import numpy as np
import numpy.typing as npt
import tifffile

__all__ = ["read_images", "save_tiff"]

Vec3i = Tuple[int, int, int]
RE_TERAFLY_ROOT = re.compile(r"^RES\((\d+)x(\d+)x(\d+)\)$")
RE_TERAFLY_NAME = re.compile(r"^\d+(_\d+)?(_\d+)?")

UINT_MAX = {
    np.dtype(np.uint8): (2**8) - 1,
    np.dtype(np.uint16): (2**16) - 1,
    np.dtype(np.uint32): (2**32) - 1,
    np.dtype(np.uint64): (2**64) - 1,
}


class ImageStack(ABC):
    """Image stack."""

    # fmt: off
    @overload
    @abstractmethod
    def __getitem__(self, key: Vec3i) -> np.float32: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: npt.NDArray[np.integer[Any]]) -> np.float32: ...
    @overload
    @abstractmethod
    def __getitem__(self, key: slice | Tuple[slice, slice] | Tuple[slice, slice, slice]) -> npt.NDArray[np.float32]: ...
    @abstractmethod
    def __getitem__(self, key): raise NotImplementedError()
    # fmt: on

    def get_full(self) -> npt.NDArray[np.float32]:
        """Get full image stack.

        Notes
        -----
        this will load the full image stack into memory.
        """
        return self[:, :, :]

    @property
    def shape(self) -> Vec3i:
        raise NotImplementedError()


def read_images(fname: str, **kwargs) -> ImageStack:
    """Read image stack."""
    if (ext := os.path.splitext(fname)[-1]) in [".tif", ".tiff"]:
        return TiffImageStack(fname, **kwargs)
    if ext == ".nrrd":
        return NrrdImageStack(fname, **kwargs)
    if TeraflyImageStack.is_root(fname):
        return TeraflyImageStack(fname, **kwargs)
    raise NotImplementedError()


def save_tiff(
    imgs: npt.NDArray,
    fname: str,
    swap_xy: bool = False,
    dtype: np.unsignedinteger | None = None,
    **kwargs,
) -> None:
    """Save image stack as tiff."""
    if swap_xy:
        imgs = imgs.swapaxes(0, 1)  # (x, y, _) -> (y, x, _)

    frames = np.rollaxis(imgs, -1)  # (_, _, z) -> (z, _, _)

    if dtype is not None:
        if np.issubdtype(frames.dtype, np.floating):
            frames *= UINT_MAX[np.dtype(dtype)]

        frames = frames.astype(dtype)

    tifffile.imwrite(
        fname,
        frames,
        photometric="minisblack",
        resolution=(1, 1),
        metadata={
            "unit": "um",
            "axes": "ZXY" if not swap_xy else "ZYX",
        },
        compression="zlib",
        compressionargs={"level": 6},
        **kwargs,
    )


class NDArrayImageStack(ImageStack):
    """NDArray image stack warpper."""

    def __init__(
        self, imgs: npt.NDArray[Any], swap_xy: bool = False, filp_xy: bool = False
    ) -> None:
        super().__init__()

        sclar_factor = 1.0
        if np.issubdtype((dtype := imgs.dtype), np.unsignedinteger):
            sclar_factor /= UINT_MAX[dtype]

        if swap_xy:
            imgs = imgs.swapaxes(0, 1)  # (y, x, _) -> (x, y, _)

        if filp_xy:
            imgs = np.flip(imgs, (0, 1))

        self.imgs = imgs.astype(np.float32) * sclar_factor

    def __getitem__(self, key):
        return self.imgs.__getitem__(key)

    def get_full(self) -> npt.NDArray[np.float32]:
        return self.imgs

    @property
    def shape(self) -> Vec3i:
        return self.imgs.shape


class TiffImageStack(NDArrayImageStack):
    """Tiff image stack warpper."""

    def __init__(
        self, fname: str, swap_xy: bool = False, filp_xy: bool = False, **kwargs
    ) -> None:
        frames = tifffile.imread(fname, **kwargs)
        imgs = np.moveaxis(frames, 0, -1)  # (z, _, _) -> (_, _, z)
        super().__init__(imgs, swap_xy=swap_xy, filp_xy=filp_xy)


class NrrdImageStack(NDArrayImageStack):
    """Nrrd image stack warpper."""

    def __init__(
        self, fname: str, swap_xy: bool = False, filp_xy: bool = False, **kwargs
    ) -> None:
        imgs, header = nrrd.read(fname, **kwargs)
        super().__init__(imgs, swap_xy=swap_xy, filp_xy=filp_xy)
        self.header = header


class TeraflyImageStack(ImageStack):
    """TeraFly image stack warpper.

    Bria, A., Iannello, G., Onofri, L. et al. TeraFly: real-time three-
    dimensional visualization and annotation of terabytes of
    multidimensional volumetric images. Nat Methods 13, 192–194 (2016).
    https://doi.org/10.1038/nmeth.3767
    """

    def __init__(self, src: str) -> None:
        super().__init__()
        self.root = src
        self.res, self.res_dirs, self.res_patch_sizes = self.get_resolutions(src)

    def __getitem__(self, key):
        """Get images in max resolution

        Examples
        --------
        ```python
        imgs[0, 0, 0]               # get value
        imgs[0:100, 0:100, 0:100]   # get range
        ```
        """

        if isinstance(key[0], slice):
            slices = [k.indices(self.res[-1][i]) for i, k in enumerate(key)]
            starts, ends, strides = np.array(slices).transpose()
            return self.get_range(starts, ends, strides)

        offset = [key[i] for i in range(3)]
        return self.get_range(offset, np.add(offset, 1)).item()

    def get_range(
        self, starts, ends, strides: int | Vec3i = 1, res_level=-1
    ) -> npt.NDArray[np.float32]:
        if isinstance(strides, int):
            strides = (strides, strides, strides)

        assert np.equal(strides, [1, 1, 1]).all()  #  TODO: support stride

        starts, ends = np.array(starts), np.array(ends)
        self._check_params(res_level, starts, np.subtract(ends, 1))

        out = np.zeros(ends - starts, dtype=np.float32)
        self._get_range(starts, ends, res_level, out=out)
        return out

    def find_correspond_imgs(self, p, res_level=-1):
        """Find the image which contain this point.

        Returns
        -------
        patch : array
        patch_offset : (int, int, int)
        """
        p = np.array(p)
        self._check_params(res_level, p)
        return self._find_correspond_imgs(p, res_level)

    def get_correspond_coord(self, p, in_res_level: int, out_res_level: int):
        raise NotImplementedError()  # TODO

    @property
    def shape(self) -> Vec3i:
        return tuple(self.res[-1])

    @classmethod
    def get_resolutions(cls, root: str) -> Tuple[List[Vec3i], List[str], List[Vec3i]]:
        """Get all resolutions.

        Returns
        -------
        resolutions : List of (int, int, int)
            Sequence of sorted resolutions (from small to large).
        roots : List[str]
            Sequence of root of resolutions respectively.
        patch_sizes : List of (int, int, int)
            Sequence of patch size of resolutions respectively.
        """

        roots = list(cls.get_resolution_dirs(root))
        assert len(roots) > 0, "no resolution detected"

        res = [RE_TERAFLY_ROOT.search(d) for d in roots]
        res = [[int(a) for a in d.groups()] for d in res if d is not None]
        res = np.array(res)
        res[:, [0, 1]] = res[:, [1, 0]]  # (y, x, _) -> (x, y, _)

        def listdir(d: str):
            return filter(RE_TERAFLY_NAME.match, os.listdir(d))

        def get_patch_size(src: str):
            y0 = next(listdir(src))
            x0 = next(listdir(os.path.join(src, y0)))
            z0 = next(listdir(os.path.join(src, y0, x0)))
            patch = read_images(os.path.join(src, y0, x0, z0))
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
        return any(RE_TERAFLY_ROOT.match(d) for d in os.listdir(root))

    @staticmethod
    def get_resolution_dirs(root: str) -> Iterable[str]:
        return filter(RE_TERAFLY_ROOT.match, os.listdir(root))

    def _check_params(self, res_level, *coords):
        assert res_level == -1  # TODO: support multi-resolutions

        res_level = len(self.res) + res_level if res_level < 0 else res_level
        assert 0 <= res_level < len(self.res), "invalid resolution level"

        res = self.res[res_level]
        for p in coords:
            assert np.less(
                [0, 0, 0], p
            ).all(), f"indices ({p[0]}, {p[1]}, {p[2]}) out of range (0, 0, 0)"

            assert np.greater(
                res, p
            ).all(), f"indices ({p[0]}, {p[1]}, {p[2]}) out of range ({res[0]}, {res[1]}, {res[2]})"

    def _get_range(self, starts, ends, res_level, out):
        # pylint: disable=too-many-locals
        shape = ends - starts
        patch, offset = self._find_correspond_imgs(starts, res_level=res_level)
        if patch is not None:
            coords = starts - offset
            lens = np.min([patch.shape - coords, shape], axis=0)
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
            dirs = [d for d in os.listdir(cur) if RE_TERAFLY_NAME.match(d)]
            diff = np.array([get_v(d) for d in dirs])
            if (invalid := diff > 10 * v).all():
                return None, None

            diff[invalid] = np.NINF  # remove values which greate than v

            # find the index of the value smaller than v and closest to v
            idx = np.argmax(diff)
            cur = os.path.join(cur, dirs[idx])

        patch = read_images(cur, swap_xy=True).get_full()
        name = os.path.splitext(os.path.basename(cur))[0]
        offset = [int(int(i) / 10) for i in name.split("_")]
        offset[0], offset[1] = offset[1], offset[0]  # (y, x, _) -> (x, y, _)
        if np.less_equal(np.add(offset, patch.shape), p).any():
            return None, None

        return patch, offset
