
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Painter utils."""

import os
import weakref
from typing import Any, Literal

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend

from swcgeom.core import SWCLike, Tree
from swcgeom.utils import (
    CameraOptions,
    SimpleCamera,
    draw_direction_indicator,
    draw_lines,
    get_fig_ax,
    palette,
)

__all__ = ["draw"]

Positions = Literal["lt", "lb", "rt", "rb"] | tuple[float, float]
locations: dict[Literal["lt", "lb", "rt", "rb"], tuple[float, float]] = {
    "lt": (0.10, 0.90),
    "lb": (0.10, 0.10),
    "rt": (0.90, 0.90),
    "rb": (0.90, 0.10),
}

ax_weak_memo = weakref.WeakKeyDictionary[Axes, dict[str, Any]]({})


def draw(
    swc: SWCLike | str,
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    show: bool | None = None,
    camera: CameraOptions = "xy",
    color: dict[int, str] | str | None = None,
    label: str | bool = True,
    direction_indicator: Positions | Literal[False] = "rb",
    unit: str | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    r"""Draw neuron tree.

    Args:
        swc: The swc tree to draw.
            If it is str, then it is treated as the path of swc file.
        fig: The figure to plot on.
        ax: The axes to plot on.
        show: Weather to call `plt.show()`.
            If not specified, it will depend on if ax is passed in, it will not be
            called, otherwise it will be called by default.
        camera: Camera options (position, look-at, up).
            One, two, or three vectors are supported, if only one vector, then threat
            it as look-at, so camera is ((0, 0, 0), look-at, (0, 1, 0)); if two vector,
            then then threat it as (look-at, up), so camera is ((0, 0, 0), look-at, up).
            An easy way is to use the presets "xy", "yz" and "zx".
        color: Color map.
            If is dict, segments will be colored by the type of parent node.If is
            string, the value will be use for any type.
        label: Label of legend, disable if False.
        direction_indicator: Draw a xyz direction indicator.
            Can be place on 'lt', 'lb', 'rt', 'rb', or custom position.
        unit: str, optional
            Add unit text, e.g.: r"$\mu m$".
        **kwargs: dict[str, Unknown]
            Forwarded to `~matplotlib.collections.LineCollection`.

    Returns:
        fig: The figure to plot on.
        ax: The axes to plot on.
    """
    # pylint: disable=too-many-locals
    swc = Tree.from_swc(swc) if isinstance(swc, str) else swc

    show = (show is True) or (show is None and ax is None)
    fig, ax = get_fig_ax(fig, ax)

    my_camera = SimpleCamera.from_options(camera)
    my_color = get_ax_color(ax, swc, color)

    xyz = swc.xyz()
    starts, ends = swc.id()[1:], swc.pid()[1:]
    lines = np.stack([xyz[starts], xyz[ends]], axis=1)
    collection = draw_lines(ax, lines, camera=my_camera, color=my_color, **kwargs)

    ax.autoscale()
    _set_ax_memo(ax, swc, label=label, handle=collection)

    if len(get_ax_swc(ax)) == 1:
        ax.set_aspect(1)
        ax.spines[["top", "right"]].set_visible(False)
        if direction_indicator is not False:
            loc = (
                locations[direction_indicator]
                if isinstance(direction_indicator, str)
                else direction_indicator
            )
            draw_direction_indicator(ax, camera=my_camera, loc=loc)
        if unit is not None:
            ax.text(0.05, 0.95, unit, transform=ax.transAxes)
    else:
        set_ax_legend(ax, loc="upper right")  # enable legend

    if show:
        fig.show(warn=False)

    return fig, ax


def get_ax_swc(ax: Axes) -> list[SWCLike]:
    ax_weak_memo.setdefault(ax, {})
    return ax_weak_memo[ax]["swc"]


def get_ax_color(
    ax: Axes,
    swc: SWCLike,
    color: dict[int, str] | str | None = None,  # TODO: improve typing
) -> str | list[str]:
    if color == "vaa3d":
        color = palette.vaa3d
    elif isinstance(color, str):
        return color  # user specified

    # choose default
    ax_weak_memo.setdefault(ax, {})
    ax_weak_memo[ax].setdefault("color", -1)
    ax_weak_memo[ax]["color"] += 1
    c = palette.default[ax_weak_memo[ax]["color"] % len(palette.default)]

    if isinstance(color, dict):
        types = swc.type()[:-1]  # colored by type of parent node
        return list(map(lambda type: color.get(type, c), types))

    return c


def set_ax_legend(ax: Axes, *args, **kwargs) -> Legend | None:
    labels = ax_weak_memo[ax].get("labels", [])
    handles = ax_weak_memo[ax].get("handles", [])

    # filter `label = False`
    handles = [a for i, a in enumerate(handles) if labels[i] is not False]
    labels = [a for a in labels if a is not False]

    if len(labels) == 0:
        return None

    return ax.legend(handles, labels, *args, **kwargs)


def _set_ax_memo(
    ax: Axes, swc: SWCLike, label: str | bool | None = None, handle: Any | None = None
):
    ax_weak_memo.setdefault(ax, {})
    ax_weak_memo[ax].setdefault("swc", [])
    ax_weak_memo[ax]["swc"].append(swc)

    if label is not None:
        label = os.path.basename(swc.source) if label is True else label
        ax_weak_memo[ax].setdefault("labels", [])
        ax_weak_memo[ax]["labels"].append(label)

    if handle is not None:
        ax_weak_memo[ax].setdefault("handles", [])
        ax_weak_memo[ax]["handles"].append(handle)
