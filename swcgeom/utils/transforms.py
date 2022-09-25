"""3D geometry transformations."""

import numpy as np
import numpy.typing as npt


__all__ = [
    "to_homogeneous",
    "angle",
    "scale3d",
    "translate3d",
    "rotate3d",
    "rotate3d_x",
    "rotate3d_y",
    "rotate3d_z",
]


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

    ones = np.full([xyz.shape[0]], fill_value=w)
    xyz4 = np.concatenate([xyz, ones], axis=1)
    return xyz4


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
    N = np.array(
        [
            [0, -nz, ny],
            [nz, 0, -nx],
            [-ny, nx, 0],
        ],
        dtype=np.float32,
    )  # pylint: disable=invalid-name

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
