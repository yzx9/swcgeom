"""Rendering related utils."""

from functools import cached_property
from typing import Dict, Literal, Tuple, cast

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from swcgeom.utils.transforms import (
    Vec3f,
    model_view_transformation,
    orthographic_projection_simple,
)

__all__ = ["CameraOptions", "Camera", "SimpleCamera", "palette"]

CameraOption = Vec3f | Tuple[Vec3f, Vec3f] | Tuple[Vec3f, Vec3f, Vec3f]
CameraPreset = Literal["xy", "yz", "zx", "yx", "zy", "xz"]
CameraPresets: Dict[CameraPreset, Tuple[Vec3f, Vec3f, Vec3f]] = {
    "xy": ((0.0, 0.0, 0.0), (+0.0, +0.0, -1.0), (+0.0, +1.0, +0.0)),
    "yz": ((0.0, 0.0, 0.0), (-1.0, +0.0, +0.0), (+0.0, +0.0, +1.0)),
    "zx": ((0.0, 0.0, 0.0), (+0.0, -1.0, +0.0), (+1.0, +0.0, +0.0)),
    "yx": ((0.0, 0.0, 0.0), (+0.0, +0.0, -1.0), (+0.0, -1.0, +0.0)),
    "zy": ((0.0, 0.0, 0.0), (-1.0, +0.0, +0.0), (+0.0, +0.0, -1.0)),
    "xz": ((0.0, 0.0, 0.0), (+0.0, -1.0, +0.0), (-1.0, +0.0, +0.0)),
}
CameraOptions = CameraOption | CameraPreset


class Camera:
    _position: Vec3f
    _look_at: Vec3f
    _up: Vec3f

    # fmt: off
    @property
    def position(self) -> Vec3f:    return self._position
    @property
    def look_at(self) -> Vec3f:     return self._look_at
    @property
    def up(self) -> Vec3f:          return self._up

    @property
    def MV(self) -> npt.NDArray[np.float32]:    raise NotImplementedError()
    @property
    def P(self) -> npt.NDArray[np.float32]:     raise NotImplementedError()
    @property
    def MVP(self) -> npt.NDArray[np.float32]:   return self.P.dot(self.MV)
    # fmt: on


class SimpleCamera(Camera):
    """Simplest camera."""

    def __init__(self, position: Vec3f, look_at: Vec3f, up: Vec3f):
        self._position = position
        self._look_at = look_at
        self._up = up

    @cached_property
    def MV(self) -> npt.NDArray[np.float32]:  # pylint: disable=invalid-name
        return model_view_transformation(self.position, self.look_at, self.up)

    @cached_property
    def P(self) -> npt.NDArray[np.float32]:  # pylint: disable=invalid-name
        return orthographic_projection_simple()

    @classmethod
    def from_options(cls, camera: CameraOptions) -> Self:
        if isinstance(camera, str):
            return cls(*CameraPresets[camera])

        if len(camera) == 2:
            return cls((0, 0, 0), camera[0], camera[1])

        if isinstance(camera[0], tuple):
            return cls((0, 0, 0), cast(Vec3f, camera), (0, 1, 0))

        return cls(*cast(Tuple[Vec3f, Vec3f, Vec3f], camera))


class Palette:
    """The palette provides default and vaa3d color matching."""

    # pylint: disable=too-few-public-methods

    default: Dict[int, str]
    vaa3d: Dict[int, str]

    def __init__(self):
        default = [
            "#F596AA",  # momo,
            "#867835",  # kimirucha,
            "#E2943B",  # kuchiba,
            "#00896C",  # aotake,
            "#B9887D",  # mizugaki,
            "#2EA9DF",  # tsuyukusa,
            "#66327C",  # sumire,
            "#52433D",  # benikeshinezumi,
        ]
        self.default = dict(enumerate(default))

        vaa3d = [
            "#ffffff",  # white, 0-undefined
            "#141414",  # black, 1-soma
            "#c81400",  # red, 2-axon
            "#0014c8",  # blue, 3-dendrite
            "#c800c8",  # purple, 4-apical dendrite
            # the following is Hanchuan’s extended color. 090331
            "#00c8c8",  # cyan, 5
            "#dcc800",  # yellow, 6
            "#00c814",  # green, 7
            "#bc5e25",  # coffee, 8
            "#b4c878",  # asparagus, 9
            "#fa6478",  # salmon, 10
            "#78c8c8",  # ice, 11
            "#6478c8",  # orchid, 12
            # the following is Hanchuan’s further extended color. 111003
            "#ff80a8",  # 13
            "#80ffa8",  # 14
            "#80a8ff",  # 15
            "#a8ff80",  # 16
            "#ffa880",  # 17
            "#a880ff",  # 18
            "#000000",  # 19 # totally black. PHC, 2012-02-15
            # the following (20-275) is used for matlab heat map. 120209 by WYN
            "#000083",
        ]
        self.vaa3d = dict(enumerate(vaa3d))


palette = Palette()
