"""Create image stack from morphology."""

from typing import Any, Iterable, List, Tuple, cast

import numpy as np
import numpy.typing as npt
import tifffile

from ..core import Tree
from ..utils import SDF, SDFCompose, SDFRoundCone
from .base import Transform

__all__ = ["ToImageStack"]


class ToImageStack(Transform[Tree, npt.NDArray[np.uint8]]):
    r"""Transform tree to image stack."""

    resolution: npt.NDArray[np.float32]
    msaa: int

    def __init__(
        self,
        resolution: float | npt.ArrayLike = 1,
        msaa: int = 8,
    ) -> None:
        """Transform tree to image stack.

        Parameters
        ----------
        resolution : int | (x, y, z), default `(1, 1, 1)`
            Resolution of image stack.
        mass : int, default `8`
            Multi-sample anti-aliasing.
        """

        if isinstance(resolution, float):
            resolution = [resolution, resolution, resolution]

        self.resolution = (resolution := np.array(resolution, dtype=np.float32))
        assert tuple(resolution.shape) == (3,), "resolution shoule be vector of 3d."

        self.msaa = msaa

    def __call__(self, x: Tree) -> npt.NDArray[np.uint8]:
        """Transform tree to image stack.

        Notes
        -----
        This method loads the entire image stack into memory, so it
        ONLY works for small image stacks, use
        :meth`transform_and_save` for big image stack.
        """
        return np.concatenate(list(self.transfrom(x)), axis=0)

    def __repr__(self) -> str:
        return (
            "ToImageStack"
            + f"-resolution-{'-'.join(self.resolution)}"
            + f"-mass-{self.msaa}"
        )

    def transfrom(
        self, x: Tree, z_per_iter: int = 1
    ) -> Iterable[npt.NDArray[np.uint8]]:
        xyz, r = x.xyz(), np.stack([x.r(), x.r(), x.r()], axis=1)  # TODO: perf
        coord_min = np.floor(np.min(xyz - r, axis=0))  # TODO: snap to grid
        coord_max = np.ceil(np.max(xyz + r, axis=0))
        grids = self.get_grids(coord_min, coord_max, z_per_iter)

        for grid in grids:
            is_in = self.get_sdf(x).is_in(grid.reshape(-1, 3)).reshape(*grid.shape[:4])
            level = np.sum(is_in, axis=3, dtype=np.uint8)
            voxel = (255 / self.msaa * level).astype(np.uint8)
            frames = np.moveaxis(voxel, 2, 0)
            yield frames

    def transform_and_save(self, fname: str, x: Tree, z_per_iter: int = 1) -> None:
        self.save_tif(fname, self.transfrom(x, z_per_iter))

    def get_grids(
        self, coord_min: npt.ArrayLike, coord_max: npt.ArrayLike, z_per_iter: int = 1
    ) -> Iterable[npt.NDArray[np.float32]]:
        """Get point grid.

        Parameters
        ----------
        coord_min, coord_max: npt.ArrayLike
            Coordinates array of shape (3,).
        z_per_iter : int
            Yeild z per iter, raising this option speeds up processing,
            but consumes more memory.

        Returns
        -------
        grid : npt.NDArray[np.float32]
            Array of shape (nx, ny, z_per_iter, k, 3).
        """

        k = np.cbrt(self.msaa)
        coord_min, coord_max = np.array(coord_min), np.array(coord_max)
        assert tuple(coord_min.shape) == (3,), "coord_min shoule be vector of 3d."
        assert tuple(coord_max.shape) == (3,), "coord_max shoule be vector of 3d."

        point_grid = np.mgrid[
            coord_min[0] : coord_max[0] : self.resolution[0],
            coord_min[1] : coord_max[1] : self.resolution[1],
            coord_min[2] : coord_max[2] : self.resolution[2],
        ]  # (3, nx, ny, nz)
        point_grid = np.rollaxis(point_grid, 0, 4)  # (nx, ny, nz, 3)

        step = self.resolution / (k + 1)
        ends = self.resolution - step / 2
        inter_grid = np.mgrid[
            step[0] : ends[0] : step[0],
            step[1] : ends[1] : step[1],
            step[2] : ends[2] : step[2],
        ]  # (3, kx, ky, kz)
        inter_grid = np.rollaxis(inter_grid, 0, 4).reshape(-1, 3)  # (k, 3)

        grids = np.expand_dims(point_grid, 3).repeat(k**3, axis=3)
        grids = cast(Any, grids + inter_grid)

        for i in range(0, grids.shape[2], z_per_iter):
            yield grids[:, :, i : i + z_per_iter]

    def get_sdf(self, x: Tree) -> SDF:
        sdfs: List[SDF] = []
        T = Tuple[Tree.Node, List[SDF]]

        def collect(n: Tree.Node, pre: List[T]) -> T:
            if len(pre) == 0:
                return (n, [])

            if len(pre) == 1:
                child, sub_sdfs = pre[0]
                sub_sdfs.append(SDFRoundCone(n.xyz(), child.xyz(), n.r, child.r))
                return (n, sub_sdfs)

            for child, sub_sdfs in pre:
                sub_sdfs.append(SDFRoundCone(n.xyz(), child.xyz(), n.r, child.r))
                sdfs.append(SDFCompose(sub_sdfs))

            return (n, [])

        _, remaining_sdfs = x.traverse(leave=collect)
        if len(remaining_sdfs) > 1:
            sdfs.append(SDFCompose(remaining_sdfs))

        return SDFCompose(sdfs)

    @staticmethod
    def save_tif(
        fname: str,
        frames: Iterable[npt.NDArray[np.uint8]],
        resolution: Tuple[float, float] = (1, 1),
    ) -> None:
        with tifffile.TiffWriter(fname) as tif:
            for frame in frames:
                tif.write(
                    frame,
                    contiguous=True,
                    photometric="minisblack",
                    resolution=resolution,
                    metadata={
                        "unit": "um",
                        "axes": "ZXY",
                    },
                )
