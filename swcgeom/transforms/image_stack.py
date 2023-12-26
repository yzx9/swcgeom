"""Create image stack from morphology.

Notes
-----
All denpendencies need to be installed, try:

```sh
pip install swcgeom[all]
```
"""

import os
import re
import time
from typing import Iterable, List, Optional, Tuple

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

from swcgeom.core import Population, Tree
from swcgeom.transforms.base import Transform

__all__ = ["ToImageStack"]


class ToImageStack(Transform[Tree, npt.NDArray[np.uint8]]):
    r"""Transform tree to image stack."""

    resolution: npt.NDArray[np.float32]

    def __init__(self, resolution: float | npt.ArrayLike = 1) -> None:
        """Transform tree to image stack.

        Parameters
        ----------
        resolution : int | (x, y, z), default `(1, 1, 1)`
            Resolution of image stack.
        """

        if isinstance(resolution, float):
            resolution = [resolution, resolution, resolution]

        self.resolution = (resolution := np.array(resolution, dtype=np.float32))
        assert tuple(resolution.shape) == (3,), "resolution shoule be vector of 3d."

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
        return "ToImageStack" + f"-resolution-{'-'.join(self.resolution)}"

    def transfrom(
        self, x: Tree, verbose: bool = True
    ) -> Iterable[npt.NDArray[np.uint8]]:
        from tqdm import tqdm

        if verbose:
            print("To image stack: " + x.source)
            time_start = time.time()

        scene = self._get_scene(x)

        xyz, r = x.xyz(), x.r().reshape(-1, 1)
        coord_min = np.floor(np.min(xyz - r, axis=0))
        coord_max = np.ceil(np.max(xyz + r, axis=0))

        samplers = self._get_samplers(coord_min, coord_max)
        total = (
            ((coord_max[2] - coord_min[2]) / self.resolution[2]).astype(np.int64).item()
        )

        if verbose:
            time_end = time.time()
            print("Prepare in: ", time_end - time_start, "s")  # type: ignore

        for sampler in tqdm(samplers, total=total) if verbose else samplers:
            voxel = sampler.sample(scene)  # shoule be shape of (x, y, z, 3) and z = 1
            frame = (255 * voxel[..., 0, 0]).astype(np.uint8)
            yield frame

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

    def _get_scene(self, x: Tree) -> Scene:
        material = ColoredMaterial((1, 0, 0)).into()
        scene = ObjectsScene()
        scene.set_background((0, 0, 0))

        def leave(n: Tree.Node, children: List[Tree.Node]) -> Tree.Node:
            for c in children:
                sdf = RoundCone(_tp3f(n.xyz()), _tp3f(c.xyz()), n.r, c.r).into()
                scene.add_object(SDFObject(sdf, material).into())

            return n

        x.traverse(leave=leave)
        scene.build_bvh()
        return scene.into()

    def _get_samplers(
        self,
        coord_min: npt.NDArray,
        coord_max: npt.NDArray,
        offset: Optional[npt.NDArray] = None,
    ) -> Iterable[RangeSampler]:
        """Get Samplers.

        Parameters
        ----------
        coord_min, coord_max: npt.ArrayLike
            Coordinates array of shape (3,).
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


def _tp3f(x: npt.NDArray) -> Tuple[float, float, float]:
    """Convert to tuple of 3 floats."""
    assert len(x) == 3
    return (float(x[0]), float(x[1]), float(x[2]))
