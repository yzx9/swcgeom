"""Create image stack from morphology."""

import re
import os
import time
from typing import Any, Iterable, List, Tuple, cast

import math
import numpy as np
import numpy.typing as npt
import tifffile
from tqdm import tqdm

from ..core import Tree, Population
from ..utils import SDF, SDFCompose, SDFRoundCone
from .base import Transform

__all__ = ["ToImageStack"]


class ToImageStack(Transform[Tree, npt.NDArray[np.uint8]]):
    r"""Transform tree to image stack."""

    resolution: npt.NDArray[np.float32]
    msaa: int
    z_per_iter: int = 1

    def __init__(
        self, resolution: float | npt.ArrayLike = 1, msaa: int = 8, z_per_iter: int = 1
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
        self.z_per_iter = z_per_iter

    def __call__(self, x: Tree) -> npt.NDArray[np.uint8]:
        """Transform tree to image stack.

        Notes
        -----
        This method loads the entire image stack into memory, so it
        ONLY works for small image stacks, use
        :meth`transform_and_save` for big image stack.
        """
        return np.concatenate(list(self.transfrom(x, verbose=False)), axis=0)

    def __repr__(self) -> str:
        return (
            "ToImageStack"
            + f"-resolution-{'-'.join(self.resolution)}"
            + f"-mass-{self.msaa}"
            + f"-z-{self.z_per_iter}"
        )

    def transfrom(
        self, x: Tree, verbose: bool = True
    ) -> Iterable[npt.NDArray[np.uint8]]:
        if verbose:
            print("To image stack: " + x.source)
            time_start = time.time()

        sdf = self.get_sdf(x)

        xyz, r = x.xyz(), np.stack([x.r(), x.r(), x.r()], axis=1)  # TODO: perf
        coord_min = np.floor(np.min(xyz - r, axis=0))  # TODO: snap to grid
        coord_max = np.ceil(np.max(xyz + r, axis=0))
        grids, total = self.get_grids(coord_min, coord_max)

        if verbose:
            time_end = time.time()
            print("Prepare in: ", time_end - time_start, "s")  # type: ignore

        for grid in tqdm(grids, total=total) if verbose else grids:
            is_in = sdf.is_in(grid.reshape(-1, 3)).reshape(*grid.shape[:4])
            level = np.sum(is_in, axis=3, dtype=np.uint8)
            voxel = (255 / self.msaa * level).astype(np.uint8)
            for i in range(voxel.shape[2]):
                yield voxel[:, :, i]

    def transform_and_save(self, fname: str, x: Tree, verbose: bool = True) -> None:
        self.save_tif(fname, self.transfrom(x, verbose=verbose))

    def transform_population(
        self, population: Population | str, verbose: bool = True
    ) -> None:
        if isinstance(population, str):
            population = Population.from_swc(population)

        # TODO: multiprocess
        for tree in population:
            tif = re.sub(r".swc$", ".tif", tree.source)
            if not os.path.isfile(tif):
                self.transform_and_save(tif, tree, verbose=verbose)

    def get_grids(
        self, coord_min: npt.ArrayLike, coord_max: npt.ArrayLike
    ) -> Tuple[Iterable[npt.NDArray[np.float32]], int]:
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

        return (
            grids[:, :, i : i + self.z_per_iter]
            for i in range(0, grids.shape[2], self.z_per_iter)
        ), math.ceil(grids.shape[2] / self.z_per_iter)

    def get_sdf(self, x: Tree) -> SDF:
        T = Tuple[Tree.Node, List[SDF], SDF | None]

        def collect(n: Tree.Node, pre: List[T]) -> T:
            if len(pre) == 0:
                return (n, [], None)

            if len(pre) == 1:
                child, sub_sdfs, last = pre[0]
                sub_sdfs.append(SDFRoundCone(n.xyz(), child.xyz(), n.r, child.r))
                return (n, sub_sdfs, last)

            sdfs: List[SDF] = []
            for child, sub_sdfs, last in pre:
                sub_sdfs.append(SDFRoundCone(n.xyz(), child.xyz(), n.r, child.r))
                sdfs.append(SDFCompose.compose(sub_sdfs))
                if last is not None:
                    sdfs.append(last)

            return (n, [], SDFCompose.compose(sdfs))

        _, sdfs, last = x.traverse(leave=collect)
        if len(sdfs) != 0:
            sdf = SDFCompose.compose(sdfs)
            if last is not None:
                sdf = SDFCompose.compose([sdf, last])
        elif last is not None:
            sdf = last
        else:
            raise ValueError("empty tree")

        return sdf

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
