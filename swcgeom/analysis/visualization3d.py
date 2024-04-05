"""Painter utils.

Notes
-----
This is a experimental function, it may be changed in the future.
"""

from typing import Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from swcgeom.analysis.visualization import (
    _set_ax_memo,
    get_ax_color,
    get_ax_swc,
    set_ax_legend,
)
from swcgeom.core import SWCLike, Tree
from swcgeom.utils.plotter_3d import draw_lines_3d

__all__ = ["draw3d"]


# TODO: support Camera
def draw3d(
    swc: SWCLike | str,
    *,
    ax: Axes,
    show: bool | None = None,
    color: Optional[dict[int, str] | str] = None,
    label: str | bool = True,
    **kwargs,
) -> tuple[Figure, Axes]:
    r"""Draw neuron tree.

    Parameters
    ----------
    swc : SWCLike | str
        If it is str, then it is treated as the path of swc file.
    fig : ~matplotlib.axes.Figure, optional
    ax : ~matplotlib.axes.Axes, optional
    show : bool | None, default `None`
        Wheather to call `plt.show()`. If not specified, it will depend
        on if ax is passed in, it will not be called, otherwise it will
        be called by default.
    color : dict[int, str] | "vaa3d" | str, optional
        Color map. If is dict, segments will be colored by the type of
        parent node.If is string, the value will be use for any type.
    label : str | bool, default True
        Label of legend, disable if False.
    **kwargs : dict[str, Unknown]
        Forwarded to `~mpl_toolkits.mplot3d.art3d.Line3DCollection`.
    """

    assert isinstance(ax, Axes3D), "only support 3D axes."

    swc = Tree.from_swc(swc) if isinstance(swc, str) else swc

    show = (show is True) or (show is None and ax is None)
    my_color = get_ax_color(ax, swc, color)  # type: ignore

    xyz = swc.xyz()
    starts, ends = swc.id()[1:], swc.pid()[1:]
    lines = np.stack([xyz[starts], xyz[ends]], axis=1)
    collection = draw_lines_3d(ax, lines, color=my_color, **kwargs)

    min_vals = lines.reshape(-1, 3).min(axis=0)
    max_vals = lines.reshape(-1, 3).max(axis=0)
    ax.set_xlim(min_vals[0], max_vals[0])
    ax.set_ylim(min_vals[1], max_vals[1])
    ax.set_zlim(min_vals[2], max_vals[2])

    _set_ax_memo(ax, swc, label=label, handle=collection)

    if len(get_ax_swc(ax)) == 1:
        # ax.set_aspect(1)
        ax.spines[["top", "right"]].set_visible(False)
    else:
        set_ax_legend(ax, loc="upper right")  # enable legend

    fig = ax.figure
    return fig, ax  # type: ignore
