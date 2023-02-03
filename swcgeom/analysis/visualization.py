"""Painter utils."""

import os
import weakref
from typing import Any, Dict, Literal, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..core import SWCLike, Tree
from ..utils import (
    draw_lines,
    draw_xyz_axes,
    get_fig_ax,
    palette,
    Camera,
    Vector3D,
)

__all__ = ["draw"]

CameraPreset = Literal["xy", "yz", "zx", "yx", "zy", "xz"]
CameraPresets: Dict[CameraPreset, Camera] = {
    "xy": Camera((0, 0, 0), (0, 0, -1), (0, 1, 0)),
    "yz": Camera((0, 0, 0), (-1, 0, 0), (0, 0, 1)),
    "zx": Camera((0, 0, 0), (0, -1, 0), (1, 0, 0)),
    "yx": Camera((0, 0, 0), (0, 0, -1), (0, -1, 0)),
    "zy": Camera((0, 0, 0), (-1, 0, 0), (0, 0, -1)),
    "xz": Camera((0, 0, 0), (0, -1, 0), (-1, 0, 0)),
}
CameraOptions = (
    Vector3D | Tuple[Vector3D, Vector3D] | Tuple[Vector3D, Vector3D, Vector3D]
)

ax_weak_dict = weakref.WeakKeyDictionary[Axes, Dict[str, Any]]({})


def draw(
    swc: SWCLike | str,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    camera: CameraOptions | CameraPreset = "xy",
    color: Dict[int, str] | str | None = None,
    label: str | Literal[True] = True,  # TODO: support False
    **kwargs,
) -> tuple[Figure, Axes]:
    """Draw neuron tree.

    Parameters
    ----------
    swc : SWCLike | str
        If it is str, then it is treated as the path of swc file.
    fig : ~matplotlib.axes.Figure, optional
    ax : ~matplotlib.axes.Axes, optional
    camera : CameraOptions | CameraPreset, default "xy"
        Camera options (position, look-at, up). One, two, or three
        vectors are supported, if only one vector, then threat it as
        look-at, so camera is ((0, 0, 0), look-at, (0, 1, 0));if two
        vector, then then threat it as (look-at, up), so camera is
        ((0, 0, 0), look-at, up). An easy way is to use the presets
        "xy", "yz" and "zx".
    color : Dict[int, str] | "vaa3d" | str, optional
        Color map. If is dict, segments will be colored by the type of
        parent node.If is string, the value will be use for any type.
    label : str | bool, default True
        Label of legend, disable if False.
    **kwargs : dict[str, Unknown]
        Forwarded to `~matplotlib.collections.LineCollection`.

    Returns
    -------
    fig : ~matplotlib.axes.Figure
    ax : ~matplotlib.axes.Axes
    """

    fig, ax = get_fig_ax(fig, ax)
    ax_weak_dict.setdefault(ax, {})
    ax_weak_dict[ax].setdefault("swc", [])

    if isinstance(swc, str):
        swc = Tree.from_swc(swc)
    ax_weak_dict[ax]["swc"].append(swc)

    my_camera = get_camera(camera)
    my_color = get_color(ax, swc, color)

    xyz = swc.xyz()
    starts, ends = swc.id()[1:], swc.pid()[1:]
    lines = np.stack([xyz[starts], xyz[ends]], axis=1)

    collection = draw_lines(lines, ax=ax, camera=my_camera, color=my_color, **kwargs)

    # legend
    ax_weak_dict[ax].setdefault("handles", [])
    ax_weak_dict[ax]["handles"].append(collection)
    set_lable(ax, swc, label)

    ax.autoscale()

    if len(ax_weak_dict[ax]["swc"]) == 1:
        ax.set_aspect(1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.05, 0.95, r"$\mu m$", transform=ax.transAxes)
        draw_xyz_axes(ax=ax, camera=my_camera)
    else:
        # legend
        handles = ax_weak_dict[ax].get("handles", [])
        labels = ax_weak_dict[ax].get("labels", [])
        ax.legend(handles, labels, loc="upper right")

    return fig, ax


def get_camera(camera: CameraOptions | CameraPreset) -> Camera:
    if isinstance(camera, str):
        return CameraPresets[camera]

    if len(camera) == 1:
        return Camera((0, 0, 0), camera, (0, 1, 0))

    if len(camera) == 2:
        return Camera((0, 0, 0), camera[0], camera[1])

    return Camera(*camera)


def get_color(
    ax: Axes, swc: SWCLike, color: Dict[int, str] | str | None
) -> str | list[str]:
    if color == "vaa3d":
        color = palette.vaa3d

    if isinstance(color, str):
        return color

    # choose default
    ax_weak_dict[ax].setdefault("color", -1)
    ax_weak_dict[ax]["color"] += 1
    c = palette.default[ax_weak_dict[ax]["color"] % len(palette.default)]

    if isinstance(color, dict):
        types = swc.type()[:-1]  # colored by type of parent node
        return list(map(lambda type: color.get(type, c), types))

    return c


def set_lable(ax: Axes, swc: SWCLike, label: str | bool):
    ax_weak_dict[ax].setdefault("labels", [])
    if label is False:
        ax_weak_dict[ax]["labels"].append(False)
        return

    if label is True:
        try:
            (_, tail) = os.path.split(swc.source)
            label = tail
        except:  # type: ignore
            label = swc.source

    ax_weak_dict[ax]["labels"].append(label)
