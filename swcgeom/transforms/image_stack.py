"""Create image stack from morphology."""

from typing import Any, List, Literal, cast

import numpy as np
import numpy.typing as npt
import skimage.io

from ..core import Tree
from ..utils import SDF, SDFCompose, SDFRoundCone
from .base import Transform

__all__ = ["ToImageStack"]


class ToImageStack(Transform[Tree, npt.NDArray[np.uint8]]):
    r"""Transform tree to image stack."""

    resolution: npt.NDArray[np.float32]
    k: int

    def __init__(
        self,
        resolution: float | npt.ArrayLike = 1,
        level: int = 9,
    ) -> None:
        """Transform tree to image stack.

        Parameters
        ----------
        resolution : int | (x, y, z), default `(1, 1, 1)`
            Resolution of image stack.
        level : int, default `9`
            Image light level.
        """

        if isinstance(resolution, float):
            resolution = [resolution, resolution, resolution]

        self.resolution = (resolution := np.array(resolution, dtype=np.float32))
        assert tuple(resolution.shape) == (3,), "resolution shoule be vector of 3d."

        self.k = np.cbrt(level - 1)

    def __call__(self, x: Tree) -> npt.NDArray[np.uint8]:
        xyz = x.xyz()
        coord_min = np.floor(xyz.min(axis=0))  # TODO: snap to grid
        coord_max = np.ceil(xyz.max(axis=0))
        grid = self.get_grid(coord_min, coord_max)

        sdf = self.get_sdf(x)
        is_in = sdf.is_in(grid.reshape(-1, 3)).reshape(*grid.shape[:4])
        level = np.sum(is_in, axis=3, dtype=np.uint8)
        voxel = (255 / (self.k**3) * level).astype(np.uint8)
        return voxel

    def __repr__(self) -> str:
        return (
            "ToImageStack"
            + f"-resolution-{'-'.join(self.resolution)}"
            + f"-level-{1+self.k**3}"
        )

    def get_grid(
        self, coord_min: npt.ArrayLike, coord_max: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """Get point grid.

        Returns
        -------
        grid : npt.NDArray[np.float32]
            Array of shape (nx, ny, nz, k, 3).
        """

        coord_min, coord_max = np.array(coord_min), np.array(coord_max)
        assert tuple(coord_min.shape) == (3,), "coord_min shoule be vector of 3d."
        assert tuple(coord_max.shape) == (3,), "coord_max shoule be vector of 3d."

        point_grid = np.mgrid[
            coord_min[0] : coord_max[0] : self.resolution[0],
            coord_min[1] : coord_max[1] : self.resolution[1],
            coord_min[2] : coord_max[2] : self.resolution[2],
        ]  # (3, nx, ny, nz)
        point_grid = np.rollaxis(point_grid, 0, 4)  # (nx, ny, nz, 3)

        step = self.resolution / (self.k + 1)
        ends = self.resolution - step / 2
        inter_grid = np.mgrid[
            step[0] : ends[0] : step[0],
            step[1] : ends[1] : step[1],
            step[2] : ends[2] : step[2],
        ]  # (3, kx, ky, kz)
        inter_grid = np.rollaxis(inter_grid, 0, 4).reshape(-1, 3)  # (k, 3)

        grid = np.expand_dims(point_grid, 3).repeat(self.k**3, axis=3)
        grid = grid + inter_grid
        return cast(Any, grid)

    def get_sdf(self, x: Tree) -> SDF:
        sdfs: List[SDF] = []

        def collect(n: Tree.Node, parent: Tree.Node | None) -> Tree.Node:
            if parent is not None:
                sdfs.append(SDFRoundCone(parent.xyz(), n.xyz(), parent.r, n.r))

            return n

        x.traverse(enter=collect)
        return SDFCompose(sdfs)

    @staticmethod
    def save_tif(
        fname: str,
        voxel: npt.NDArray[np.uint8],
        stack_axis: Literal["x", "y", "z"] = "z",
    ) -> None:
        if stack_axis == "x":
            images = voxel
        elif stack_axis == "y":
            images = np.moveaxis(voxel, 1, 0)
        else:
            images = np.moveaxis(voxel, 2, 0)

        skimage.io.imsave(fname, images, plugin="tifffile")
