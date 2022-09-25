"""Painter utils."""


from typing import Dict, Literal, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import SWCLike, Tree
from ..utils import draw_lines, draw_xyz_axes, get_fig_ax, palette, Camera, Vector3D

__all__ = ["draw"]

DEFAULT_COLOR = palette.momo
CAMERA_PRESET = Literal[  # pylint: disable=invalid-name
    "xy", "yz", "zx", "yx", "zy", "xz"
]
CAMERA_PRESETS: Dict[CAMERA_PRESET, Camera] = {
    "xy": Camera((0, 0, 0), (0, 0, -1), (0, 1, 0)),
    "yz": Camera((0, 0, 0), (-1, 0, 0), (0, 0, 1)),
    "zx": Camera((0, 0, 0), (0, -1, 0), (1, 0, 0)),
    "yx": Camera((0, 0, 0), (0, 0, -1), (0, -1, 0)),
    "zy": Camera((0, 0, 0), (-1, 0, 0), (0, 0, -1)),
    "xz": Camera((0, 0, 0), (0, -1, 0), (-1, 0, 0)),
}
CAMERA_OPTIONS = (
    Vector3D | Tuple[Vector3D, Vector3D] | Tuple[Vector3D, Vector3D, Vector3D]
)


def draw(
    swc: SWCLike | str,
    *,
    color: Dict[int, str] | str | None = None,
    first: bool = True,
    fig: Figure | None = None,
    ax: Axes | None = None,
    camera: CAMERA_OPTIONS | CAMERA_PRESET = "xy",
    **kwargs,
) -> tuple[Figure, Axes]:
    """Draw neuron tree.

    Parameters
    ----------
    swc : SWCLike | str
        If it is str, then it is treated as the path of swc file.
    color : Dict[int, str] | str, optional
        Color map. If is dict, segments will be colored by the type of
        parent node.If is string, the value will be use for any type.
    ax : ~matplotlib.axes.Axes, optional
        A subplot of `~matplotlib`. If `None`, a new one will be
        created.
    first : bool, default to `True`
        If multiple neuron plotted on same axes, set to `False` on
        subsequent calls.
    camera : CameraOptions | "xy" | "yz" | "zx", default "xy"
        Camera options (position, look-at, up). One, two, or three
        vectors are supported, if only one vector, then threat it as
        look-at, so camera is ((0, 0, 0), look-at, (0, 1, 0));if two
        vector, then then threat it as (look-at, up), so camera is
        ((0, 0, 0), look-at, up). An easy way is to use the presets
        "xy", "yz" and "zx".
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.

    Returns
    -------
    ax : ~matplotlib.axes.Axes
        If provided, return as-is.
    collection : ~matplotlib.collections.LineCollection
        Drawn line collection.
    """

    if isinstance(swc, str):
        swc = Tree.from_swc(swc)

    if isinstance(color, dict):
        types = swc.type()[:-1]  # colored by type of parent node
        color_map = list(map(lambda t: color.get(t, DEFAULT_COLOR), types))
    else:
        color_map = color if color is not None else DEFAULT_COLOR

    if isinstance(camera, str):
        camera = CAMERA_PRESETS[camera]
    elif len(camera) == 1:
        camera = Camera((0, 0, 0), camera, (0, 1, 0))
    elif len(camera) == 2:
        camera = Camera((0, 0, 0), camera[0], camera[1])
    else:
        camera = Camera(*camera)

    xyz = swc.xyz()
    starts, ends = swc.id()[1:], swc.pid()[1:]
    lines = np.stack([xyz[starts], xyz[ends]], axis=1)

    fig, ax = get_fig_ax(fig, ax)
    draw_lines(lines, ax=ax, camera=camera, color=color_map, **kwargs)
    ax.autoscale()
    if first:
        ax.set_aspect(1)
        ax.spines[["top", "right"]].set_visible(False)

        ax.text(0.05, 0.95, r"$\mu m$", transform=ax.transAxes)

        draw_xyz_axes(ax=ax, camera=camera)

    return fig, ax
