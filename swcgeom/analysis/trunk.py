"""Plot trunk and florets."""

# pylint: disable=invalid-name

from itertools import chain
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Ellipse, Patch, Rectangle

from swcgeom.analysis.visualization import draw
from swcgeom.core import Tree, get_subtree, to_subtree
from swcgeom.utils import get_fig_ax, mvee

__all__ = ["draw_trunk"]

Bounds = Literal["aabb", "ellipse"]
Projection = Literal["2d"]


def draw_trunk(
    t: Tree,
    florets: Iterable[int | Iterable[int]],
    *,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    bound: Bounds | Tuple[Bounds, Dict[str, Any]] | None = "ellipse",
    point: bool | Dict[str, Any] = True,
    projection: Projection = "2d",
    cmap: Any = "viridis",
    **kwargs,
) -> Tuple[Figure, Axes]:
    """Draw trunk tree.

    Parameters
    ----------
    t : Tree
    florets : List of (int | List of int)
        The florets that needs to be removed, each floret can be a
        subtree or multiple subtrees (e.g., dendrites are a bunch of
        subtrees), each number is the id of a tree node.
    fig : ~matplotlib.figure.Figure, optional
    ax : ~matplotlib.axes.Axes, optional
    bound : Bounds | (Bounds, Dict[str, Any]) | None, default 'ellipse'
        Kind of bound, support 'aabb', 'ellipse'. If bound is None, no
        bound will be drawn. If bound is a tuple, the second item will
        used as kwargs and forward to draw function.
    point : bool | Dict[str, Any], default True
        Draw point at the start of a subtree. If point is False, no
        point will be drawn. If point is a dict, this will used a
        kwargs and forward to draw function.
    cmap : Any, default 'viridis'
        Colormap, any value supported by ~matplotlib.cm.Colormap. We
        will use the ratio of the length of the subtree to the total
        length of the tree to determine the color.
    **kwargs : Dict[str, Any]
        Forward to ~swcgeom.analysis.draw.
    """
    # pylint: disable=too-many-locals
    trunk, tss = split_florets(t, florets)
    lens = get_length_ratio(t, tss)

    cmap = cm.get_cmap(cmap)
    c = cmap(lens)

    fig, ax = get_fig_ax(fig, ax)
    if bound is not None:
        for ts, cc in zip(tss, c):
            draw_bound(ts, ax, bound, projection, color=cc)

    if point is not False:
        point_kwargs = point if isinstance(point, dict) else {}
        for ts, cc in zip(tss, c):
            draw_point(ts, ax, projection, color=cc, **point_kwargs)

    draw(trunk, ax=ax, color=cmap(1), **kwargs)
    return fig, ax


def split_florets(
    t: Tree, florets: Iterable[int | Iterable[int]]
) -> Tuple[Tree, List[List[Tree]]]:
    florets = [[i] if isinstance(i, (int, np.integer)) else i for i in florets]
    subtrees = [[get_subtree(t, ff) for ff in f] for f in florets]
    trunk = to_subtree(t, chain(*florets))
    return trunk, subtrees


def get_length_ratio(t: Tree, tss: List[List[Tree]]) -> Any:
    lens = np.array([sum(t.length() for t in ts) for ts in tss])
    return lens / t.length()


# Bounds


def draw_bound(
    ts: Iterable[Tree],
    ax: Axes,
    bound: Bounds | Tuple[Bounds, Dict[str, Any]],
    projection: Projection,
    **kwargs,
) -> None:
    kind, bound_kwargs = (bound, {}) if isinstance(bound, str) else bound
    if projection == "2d":
        patch = create_bound_2d(ts, kind, **kwargs, **bound_kwargs)
    else:
        raise ValueError(f"unsupported projection {projection}")

    ax.add_patch(patch)


def create_bound_2d(ts: Iterable[Tree], bound: Bounds, **kwargs) -> Patch:
    xyz = np.concatenate([t.xyz() for t in ts])
    xy = xyz[:, :2]  # TODO: camera

    if bound == "aabb":
        return create_aabb_2d(xy, **kwargs)
    if bound == "ellipse":
        return create_ellipse_2d(xy, **kwargs)
    raise ValueError(f"unsupport bound `{bound}` in 2d projection")


def create_aabb_2d(xy: npt.NDArray, fill: bool = False, **kwargs) -> Rectangle:
    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()
    xmax, ymax = xy[:, 0].max(), xy[:, 1].max()
    width, height = xmax - xmin, ymax - ymin
    rect = Rectangle(
        xy=(xmin, ymin), width=width, height=height, angle=0, fill=fill, **kwargs
    )
    return rect


def create_ellipse_2d(xy: npt.NDArray, fill: bool = False, **kwargs) -> Ellipse:
    ellipse = mvee(xy)
    patch = Ellipse(
        xy=ellipse.centroid,  # type:ignore
        width=ellipse.a,
        height=ellipse.b,
        angle=ellipse.alpha,
        fill=fill,
        **kwargs,
    )
    return patch


# point


def draw_point(ts: Iterable[Tree], ax: Axes, projection: Projection, **kwargs) -> None:
    if projection == "2d":
        patch = create_point_2d(ts, **kwargs)
    else:
        raise ValueError(f"unsupported projection {projection}")

    ax.add_patch(patch)


def create_point_2d(
    ts: Iterable[Tree], radius: Optional[float] = None, **kwargs
) -> Circle:
    if radius is None:
        xyz = np.concatenate([t.xyz() for t in ts])  # TODO: cache
        radius = 0.05 * min(
            xyz[:, 0].max() - xyz[:, 0].min(),
            xyz[:, 1].max() - xyz[:, 1].min(),
        )
        radius = cast(float, radius)

    center = np.mean([t.xyz()[0, :2] for t in ts], axis=0)
    return Circle(center, radius, **kwargs)


# Helpers


def get_dendrites(tree: Tree) -> Iterable[int]:
    return (t.node(0).id for t in tree.get_dendrites())
