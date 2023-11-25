"""Rendering related utils."""

from functools import cached_property
from typing import Dict, Literal, Optional, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from typing_extensions import Self

from swcgeom.utils.transforms import (
    Vec3f,
    model_view_transformation,
    orthographic_projection_simple,
    to_homogeneous,
    translate3d,
)

__all__ = [
    "CameraOptions",
    "SimpleCamera",
    "palette",
    "draw_lines",
    "draw_direction_indicator",
    "draw_circles",
    "get_fig_ax",
]

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


class SimpleCamera:
    """Simplest camera."""

    def __init__(self, position: Vec3f, look_at: Vec3f, up: Vec3f):
        self.position = position
        self.look_at = look_at
        self.up = up

    @cached_property
    def MV(self) -> npt.NDArray[np.float32]:  # pylint: disable=invalid-name
        return model_view_transformation(self.position, self.look_at, self.up)

    @cached_property
    def P(self) -> npt.NDArray[np.float32]:  # pylint: disable=invalid-name
        return orthographic_projection_simple()

    @cached_property
    def MVP(self) -> npt.NDArray[np.float32]:  # pylint: disable=invalid-name
        return self.P.dot(self.MV)

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


def draw_lines(
    ax: Axes, lines: npt.NDArray[np.floating], camera: SimpleCamera, **kwargs
) -> LineCollection:
    """Draw lines.

    Parameters
    ----------
    ax : ~matplotlib.axes.Axes
    lines : A collection of coords of lines
        Excepting a ndarray of shape (N, 2, 3), the axis-2 holds two points,
        and the axis-3 holds the coordinates (x, y, z).
    camera : Camera
        Camera position.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.
    """

    T = camera.MVP
    T = translate3d(*camera.position).dot(T)  # keep origin

    starts, ends = lines[:, 0], lines[:, 1]
    starts, ends = to_homogeneous(starts, 1), to_homogeneous(ends, 1)
    starts, ends = np.dot(T, starts.T).T[:, 0:2], np.dot(T, ends.T).T[:, 0:2]

    edges = np.stack([starts, ends], axis=1)
    return ax.add_collection(LineCollection(edges, **kwargs))  # type: ignore


def draw_direction_indicator(
    ax: Axes, camera: SimpleCamera, loc: Tuple[float, float]
) -> None:
    x, y = loc
    direction = camera.MV.dot(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [0, 0, 0, 1],
        ]
    )

    arrow_length, text_offset = 0.05, 0.05  # TODO: may still overlap
    text_colors = [["x", "red"], ["y", "green"], ["z", "blue"]]
    for (dx, dy, dz, _), (text, color) in zip(direction, text_colors):
        if 1 - abs(dz) < 1e-5:
            continue

        ax.arrow(
            x,
            y,
            arrow_length * dx,
            arrow_length * dy,
            head_length=0.02,
            head_width=0.01,
            color=color,
            transform=ax.transAxes,
        )

        ax.text(
            x + (arrow_length + text_offset) * dx,
            y + (arrow_length + text_offset) * dy,
            text,
            color=color,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )


def draw_circles(
    ax: Axes,
    x: npt.NDArray,
    y: npt.NDArray,
    *,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
    cmap: str | Colormap = "viridis",
) -> PatchCollection:
    """Draw a sequential of circles."""

    y_min = y.min() if y_min is None else y_min
    y_max = y.max() if y_max is None else y_max
    norm = Normalize(y_min, y_max)

    color_map = cmap if isinstance(cmap, Colormap) else cm.get_cmap(name=cmap)
    colors = color_map(norm(y))

    circles = [
        Circle((0, 0), xi, color=color) for xi, color in reversed(list(zip(x, colors)))
    ]
    patches = PatchCollection(circles, match_original=True)
    patches.set_cmap(color_map)
    patches.set_norm(norm)
    patches: PatchCollection = ax.add_collection(patches)  # type: ignore

    ax.set_aspect(1)
    ax.autoscale()
    return patches


def get_fig_ax(
    fig: Figure | None = None, ax: Axes | None = None
) -> Tuple[Figure, Axes]:
    if fig is None:
        fig = plt.gcf() if ax is None else ax.get_figure()

    if ax is None:
        ax = fig.gca()

    return fig, ax
