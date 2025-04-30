
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""Create image stack from morphology.

NOTE: All denpendencies need to be installed, try:

```sh
pip install swcgeom[all]
```
"""

import os
import re
import time
from collections.abc import Iterable
from typing import Sequence

import numpy as np
import numpy.typing as npt
import tifffile
from sdflit import (
    ColoredMaterial,
    ObjectsScene,
    RangeSampler,
    RoundCone,
    Scene,
    SDFObject,
)
from tqdm import tqdm
from typing_extensions import deprecated, override

from swcgeom.core import Population, Tree
from swcgeom.transforms.base import Transform

__all__ = ["ToImageStack"]


class ToImageStack(Transform[Tree, npt.NDArray[np.uint8]]):
    r"""Transform tree to image stack."""

    resolution: npt.NDArray[np.float32]

    def __init__(self, resolution: int | float | npt.ArrayLike = 1) -> None:
        """Transform tree to image stack.

        Args:
            resolution: Resolution of image stack.
                If a scalar, it will be broadcasted to a vector of 3d.
        """
        if isinstance(resolution, (int, float, np.integer, np.floating)):
            resolution = [resolution, resolution, resolution]  # type: ignore

        self.resolution = np.array(resolution, dtype=np.float32)
        assert len(self.resolution) == 3, "resolution should be vector of 3d."

    @override
    def __call__(self, x: Tree) -> npt.NDArray[np.uint8]:
        """Transform tree to image stack.

        NOTE: This method loads the entire image stack into memory, so it ONLY works
        for small image stacks, use :meth`transform_and_save` for big image stack.
        """
        return np.stack(list(self.transform(x, verbose=False)), axis=0)

    def transform(
        self,
        x: Tree | Sequence[Tree],
        verbose: bool = True,
        *,
        ranges: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    ) -> Iterable[npt.NDArray[np.uint8]]:
        trees = [x] if isinstance(x, Tree) else x
        if not trees:
            return iter([])  # Return empty iterator if sequence is empty

        time_start = None
        if verbose:
            sources = ", ".join(t.source for t in trees if t.source)
            print(f"To image stack: {sources if sources else 'unnamed trees'}")
            time_start = time.time()

        scene = self._get_scene(trees)

        if ranges is None:
            all_xyz = np.concatenate([t.xyz() for t in trees], axis=0)
            all_r = np.concatenate([t.r() for t in trees], axis=0).reshape(-1, 1)
            if all_xyz.size == 0:  # Handle empty trees
                coord_min = np.zeros(3, dtype=np.float32)
                coord_max = np.zeros(3, dtype=np.float32)
            else:
                coord_min = np.floor(np.min(all_xyz - all_r, axis=0))
                coord_max = np.ceil(np.max(all_xyz + all_r, axis=0))
        else:
            assert len(ranges) == 2
            coord_min = np.array(ranges[0])
            coord_max = np.array(ranges[1])
            assert len(coord_min) == len(coord_max) == 3

        samplers = self._get_samplers(coord_min, coord_max)

        if verbose and time_start is not None:
            total = (coord_max[2] - coord_min[2]) / self.resolution[2]
            samplers = tqdm(samplers, total=total.astype(np.int64).item())

            time_end = time.time()
            print("Prepare in: ", time_end - time_start, "s")

        for sampler in samplers:
            voxel = sampler.sample(scene)  # should be shape of (x, y, z, 3) and z = 1
            frame = (255 * voxel[..., 0, 0]).astype(np.uint8)
            yield frame

    @deprecated("Use transform instead")
    def transfrom(
        self,
        x: Tree,
        verbose: bool = True,
        *,
        ranges: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    ) -> Iterable[npt.NDArray[np.uint8]]:
        return self.transform(x, verbose, ranges=ranges)

    def transform_and_save(
        self, fname: str, x: Tree | Sequence[Tree], verbose: bool = True, **kwargs
    ) -> None:
        self.save_tif(fname, self.transform(x, verbose=verbose, **kwargs))

    def transform_population(
        self, population: Population | str, verbose: bool = True
    ) -> None:
        trees = (
            Population.from_swc(population)
            if isinstance(population, str)
            else population
        )

        if verbose:
            trees = tqdm(trees)

        # TODO: multiprocess
        for tree in trees:
            tif = re.sub(r".swc$", ".tif", tree.source)
            if not os.path.isfile(tif):
                self.transform_and_save(tif, tree, verbose=False)

    @override
    def extra_repr(self) -> str:
        res = ",".join(f"{a:.4f}" for a in self.resolution)
        return f"resolution=({res})"

    def _get_scene(self, trees: Sequence[Tree]) -> Scene:
        material = ColoredMaterial((1, 0, 0)).into()
        scene = ObjectsScene()
        scene.set_background((0, 0, 0))

        def leave(n: Tree.Node, children: list[Tree.Node]) -> Tree.Node:
            for c in children:
                sdf = RoundCone(_tp3f(n.xyz()), _tp3f(c.xyz()), n.r, c.r).into()
                scene.add_object(SDFObject(sdf, material).into())
            return n

        for tree in trees:
            tree.traverse(leave=leave)

        scene.build_bvh()
        return scene.into()

    def _get_samplers(
        self,
        coord_min: npt.NDArray,
        coord_max: npt.NDArray,
        offset: npt.NDArray | None = None,
    ) -> Iterable[RangeSampler]:
        """Get Samplers.

        Args:
            coord_min, coord_max: Coordinates array of shape (3,).
        """

        eps = 1e-6
        stride = self.resolution
        offset = offset or (stride / 2)

        xmin, ymin, zmin = _tp3f(coord_min + offset)
        xmax, ymax, zmax = _tp3f(coord_max)
        z = zmin
        while z < zmax:
            yield RangeSampler(
                (xmin, ymin, z), (xmax, ymax, z + stride[2] - eps), _tp3f(stride)
            )
            z += stride[2]

    @staticmethod
    def save_tif(
        fname: str,
        frames: Iterable[npt.NDArray[np.uint8]],
        resolution: tuple[float, float] = (1, 1),
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


def _tp3f(x: npt.NDArray) -> tuple[float, float, float]:
    """Convert to tuple of 3 floats."""
    assert len(x) == 3
    return (float(x[0]), float(x[1]), float(x[2]))
