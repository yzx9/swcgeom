"""Create image stack from morphology."""

from typing import Any, List, Literal, Tuple, cast

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
        xyz = x.xyz()
        coord_min = np.floor(xyz.min(axis=0))  # TODO: snap to grid
        coord_max = np.ceil(xyz.max(axis=0))
        grid = self.get_grid(coord_min, coord_max)

        is_in = self.get_sdf(x).is_in(grid.reshape(-1, 3)).reshape(*grid.shape[:4])
        level = np.sum(is_in, axis=3, dtype=np.int8)
        voxel = (255 / self.msaa * level).astype(np.uint8)
        return voxel

    def __repr__(self) -> str:
        return (
            "ToImageStack"
            + f"-resolution-{'-'.join(self.resolution)}"
            + f"-mass-{self.msaa}"
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

        grid = np.expand_dims(point_grid, 3).repeat(k**3, axis=3)
        grid = grid + inter_grid
        return cast(Any, grid)

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

        x.traverse(leave=collect)
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
