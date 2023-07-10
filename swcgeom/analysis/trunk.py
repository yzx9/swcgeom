"""Plot trunk and florets."""

from itertools import chain
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

import numpy as np
import numpy.linalg as la
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse, Rectangle

from ..core import SWCLike, Tree, get_subtree, to_subtree
from ..utils import get_fig_ax
from .visualization import draw

# pylint: disable=invalid-name

__all__ = ["draw_trunk"]

Bounds = Literal["aabb", "ellipse"]
Projection = Literal["2d"]


def draw_trunk(
    t: SWCLike,
    florets: Iterable[int | Iterable[int]],
    *,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    bound: Bounds | Tuple[Bounds, Dict[str, Any]] = "ellipse",
    projection: Projection = "2d",
) -> Tuple[Figure, Axes]:
    trunk, tss = split_florets(t, florets)
    fig, ax = get_fig_ax(fig, ax)
    bound_kind, bound_kwargs = (bound, {}) if isinstance(bound, str) else bound
    for ts in tss:
        draw_bound(ts, ax, bound_kind, projection, **bound_kwargs)
    draw(trunk, ax=ax)
    return fig, ax


def split_florets(
    t: SWCLike, florets: Iterable[int | Iterable[int]]
) -> Tuple[Tree, List[List[Tree]]]:
    florets = [[i] if isinstance(i, (int, np.integer)) else i for i in florets]
    subtrees = [[get_subtree(t, ff) for ff in f] for f in florets]
    trunk = to_subtree(t, chain(*florets))
    return trunk, subtrees


# Bounds


def draw_bound(
    ts: Iterable[SWCLike], ax: Axes, bound: Bounds, projection: Projection, **kwargs
) -> None:
    if projection == "2d":
        xyz = np.concatenate([t.xyz() for t in ts])
        xy = xyz[:, :2]  # TODO: camera

        if bound == "aabb":
            patch = create_aabb_2d(xy, **kwargs)
        elif bound == "ellipse":
            patch = create_ellipse_2d(xy, **kwargs)
        else:
            raise ValueError(f"unsupport bound {bound} and projection {projection}")

    else:
        raise ValueError(f"unsupported projection {projection}")

    ax.add_patch(patch)


def create_aabb_2d(xy: npt.NDArray, fill: bool = False, **kwargs) -> Rectangle:
    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()
    xmax, ymax = xy[:, 0].max(), xy[:, 1].max()
    width, height = xmax - xmin, ymax - ymin
    rect = Rectangle(
        xy=(xmin, ymin), width=width, height=height, angle=0, fill=fill, **kwargs
    )
    return rect


def create_ellipse_2d(xy: npt.NDArray, fill: bool = False, **kwargs) -> Ellipse:
    centroid, a, b, alpha = get_stand_ellipse(xy)
    ellipse = Ellipse(
        xy=centroid, width=a, height=b, angle=alpha, fill=fill, **kwargs  # type:ignore
    )
    return ellipse


# Helpers


def get_dendrites(tree: Tree) -> Iterable[int]:  # TODO: move to `Tree`
    return [n.id for n in tree.soma().children() if n.type in [3, 4]]


def mvee(
    points: npt.NDArray[np.floating], tol: float = 1e-3
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Finds the Minimum Volume Enclosing Ellipsoid.

    Returns
    -------
    A : matrix of shape (d, d)
        The ellipse equation in the 'center form': (x-c)' * A * (x-c) = 1
    centroid : array of shape (d,)
        The center coordinates of the ellipse.

    Reference
    ---------
    1. http://stackoverflow.com/questions/14016898/port-matlab-bounding-ellipsoid-code-to-python
    2. http://stackoverflow.com/questions/1768197/bounding-ellipse/1768440#1768440
    3. https://minillinim.github.io/GroopM/dev_docs/groopm.ellipsoid-pysrc.html
    """

    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N
    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, la.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = la.norm(new_u - u)
        u = new_u
    c = np.dot(u, points)
    A = (
        la.inv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(c, c))
        / d
    )
    return A, c


def get_stand_ellipse(
    ps: npt.NDArray,
) -> Tuple[Iterable[float], float, float, float]:
    A, centroid = mvee(ps)

    # V is the rotation matrix that gives the orientation of the ellipsoid.
    # https://en.wikipedia.org/wiki/Rotation_matrix
    # http://mathworld.wolfram.com/RotationMatrix.html
    _U, D, V = la.svd(A)

    # x, y radii.
    rx, ry = 1.0 / np.sqrt(D)
    # Major and minor semi-axis of the ellipse.
    dx, dy = 2 * rx, 2 * ry
    a, b = max(dx, dy), min(dx, dy)
    # Eccentricity
    # e = np.sqrt(a ** 2 - b ** 2) / a

    arcsin = -1.0 * np.rad2deg(np.arcsin(V[0][0]))
    arccos = np.rad2deg(np.arccos(V[0][1]))
    # Orientation angle (with respect to the x axis counterclockwise).
    alpha = arccos if arcsin > 0.0 else -1.0 * arccos

    return centroid, a, b, alpha
