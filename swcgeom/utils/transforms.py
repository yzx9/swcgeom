"""3D geometry transformations."""

from typing import Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "angle",
    "scale3d",
    "translate3d",
    "rotate3d",
    "rotate3d_x",
    "rotate3d_y",
    "rotate3d_z",
    "to_homogeneous",
    "model_view_trasformation",
    "orthographic_projection_simple",
]

Vector3D = Tuple[float, float, float]


def angle(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Get the agnle of vectors.

    Returns
    -------
    angle : float
        Angle in radians.
    """
    a, b = np.array(a), np.array(b)
    costheta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    theta = np.arccos(costheta)
    return theta if np.cross(a, b) > 0 else -theta


def scale3d(sx: float, sy: float, sz: float) -> npt.NDArray[np.float32]:
    """Get the 3D scale transfomation matrix.

    Returns
    -------
    T : np.NDArray
        The homogeneous transfomation matrix, shape (4, 4).
    """
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def translate3d(tx: float, ty: float, tz: float) -> npt.NDArray[np.float32]:
    """Get the 3D translate transfomation matrix.

    Returns
    -------
    T : np.NDArray
        The homogeneous transfomation matrix, shape (4, 4).
    """
    return np.array(
        [
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotate3d(n: npt.ArrayLike, theta: float) -> npt.NDArray[np.float32]:
    r"""Get the 3D rotation transfomation matrix.

    Rotate v with axis n by an angle theta according to the right hand rule,
    follow rodrigues' rotaion formula.

    .. math::

        R(\mathbf{n}, \alpha) = \cos{\alpha} \cdot \mathbf{I}
        + (1-\cos{\alpha}) \cdot \mathbf{n} \cdot \mathbf{n^T}
        + \sin{\alpha} \cdot \mathbf{N} \\

        N  = \begin{pmatrix}
        0 & -n_z & n_y \\
        n_z & 0 & -n_x \\
        -n_y & n_x & 0
        \end{pmatrix}

    Parameters
    ----------
    n : ArrayLike
        Rotation axis.
    theta : float
        Rotation angle in radians.

    Returns
    -------
    T : np.NDArray
        The homogeneous transfomation matrix, shape (4, 4).
    """

    n = np.array(n)
    nx, ny, nz = n[0:3]
    # pylint: disable-next=invalid-name
    N = np.array(
        [
            [0, -nz, ny],
            [nz, 0, -nx],
            [-ny, nx, 0],
        ],
        dtype=np.float32,
    )

    return (
        np.cos(theta) * np.identity(4)
        + (1 - np.cos(theta)) * n * n[:, None]
        + np.sin(theta) * N
    )


def rotate3d_x(theta: float) -> npt.NDArray[np.float32]:
    """Get the 3D rotation transfomation matrix.

    Rotate 3D vector `v` with `x`-axis by an angle theta according to the right
    hand rule.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    T : np.NDArray
        The homogeneous transfomation matrix, shape (4, 4).
    """

    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotate3d_y(theta: float) -> npt.NDArray[np.float32]:
    """Get the 3D rotation transfomation matrix.

    Rotate 3D vector `v` with `y`-axis by an angle theta according to the right
    hand rule.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    T : np.NDArray
        The homogeneous transfomation matrix, shape (4, 4).
    """
    return np.array(
        [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def rotate3d_z(theta: float) -> npt.NDArray[np.float32]:
    """Get the 3D rotation transfomation matrix.

    Rotate 3D vector `v` with `z`-axis by an angle theta according to the right
    hand rule.

    Parameters
    ----------
    theta : float
        Rotation angle in radians.

    Returns
    -------
    T : np.NDArray
        The homogeneous transfomation matrix, shape (4, 4).
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def to_homogeneous(xyz: npt.ArrayLike, w: float) -> npt.NDArray[np.float32]:
    """Fill xyz to homogeneous coordinates.

    Parameters
    ----------
    xyz : ArrayLike
        Coordinate of shape (N, 3)
    w : float
        w of homogeneous coordinate, 1 for dot, 0 for vector.

    Returns
    -------
    xyz4 : npt.NDArray[np.float32]
        Array of shape (N, 4)
    """
    xyz = np.array(xyz)
    if xyz.shape[1] == 4:
        return xyz

    filled = np.full((xyz.shape[0], 1), fill_value=w)
    xyz4 = np.concatenate([xyz, filled], axis=1)
    return xyz4


def model_view_trasformation(
    position: Vector3D, look_at: Vector3D, up: Vector3D
) -> npt.NDArray[np.float32]:
    r"""Play model/view transformation.

    Parameters
    ----------
    position: Tuple[float, float, float]
        Camera position \vec{e}.
    look_at: Tuple[float, float, float]
        Camera look-at \vec{g}.
    up: Tuple[float, float, float]
        Camera up direction \vec{t}.
    """

    e = np.array(position, dtype=np.float32)
    g = np.array(look_at, dtype=np.float32) / np.linalg.norm(look_at)
    t = np.array(up, dtype=np.float32) / np.linalg.norm(up)

    t_view = translate3d(*(-1 * e))
    r_view = np.array(
        [
            [*np.cross(g, t), 0],
            [*t, 0],
            [*(-1 * g), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )
    return np.dot(r_view, t_view)


def orthographic_projection_simple() -> npt.NDArray[np.float32]:
    """Simple orthographic projection by drop z-axis"""
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],  # drop z-axis
            [0, 0, 0, 0],  # drop w
        ],
        dtype=np.float32,
    )
