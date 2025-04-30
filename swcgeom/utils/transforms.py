
# SPDX-FileCopyrightText: 2022 - 2025 Zexin Yuan <pypi@yzx9.xyz>
#
# SPDX-License-Identifier: Apache-2.0

"""3D geometry transformations."""

import numpy as np
import numpy.typing as npt

__all__ = [
    "Vec3f",
    "angle",
    "scale3d",
    "translate3d",
    "rotate3d",
    "rotate3d_x",
    "rotate3d_y",
    "rotate3d_z",
    "to_homogeneous",
    "model_view_transformation",
    "orthographic_projection_simple",
]

Vec3f = tuple[float, float, float]


def angle(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Get the signed angle between two vectors.

    The angle is positive if the rotation from a to b is counter-clockwise, and
    negative if clockwise.

    >>> angle([1, 0, 0], [1, 0, 0])  # identical
    0.0
    >>> angle([1, 0, 0], [0, 1, 0])  # 90 degrees counter-clockwise
    1.5707963267948966
    >>> angle([1, 0, 0], [0, -1, 0])  # 90 degrees clockwise
    -1.5707963267948966

    Returns:
        angle: Angle in radians between -π and π.
    """

    a = np.asarray(a)
    b = np.asarray(b)

    # Normalize vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    # Calculate cosine of angle
    cos_theta = np.dot(a_norm, b_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Ensure within valid range
    theta = np.arccos(cos_theta)

    # Determine sign using cross product
    cross = np.cross(a_norm, b_norm)
    sign = np.sign(cross[2])  # Use z-component for 3D
    return float(sign * theta)


def scale3d(sx: float, sy: float, sz: float) -> npt.NDArray[np.float32]:
    """Get the 3D scale transformation matrix.

    >>> np.allclose(
    ...     scale3d(2, 3, 4),
    ...     [
    ...         [2.0, 0.0, 0.0, 0.0],
    ...         [0.0, 3.0, 0.0, 0.0],
    ...         [0.0, 0.0, 4.0, 0.0],
    ...         [0.0, 0.0, 0.0, 1.0],
    ...     ],
    ... )
    True

    Returns:
        T: The homogeneous transformation matrix, shape (4, 4).
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
    """Get the 3D translate transformation matrix.

    >>> np.allclose(
    ...     translate3d(1, 2, 3),
    ...     [
    ...         [1.0, 0.0, 0.0, 1.0],
    ...         [0.0, 1.0, 0.0, 2.0],
    ...         [0.0, 0.0, 1.0, 3.0],
    ...         [0.0, 0.0, 0.0, 1.0],
    ...     ],
    ... )
    True

    Returns:
        T: The homogeneous transformation matrix, shape (4, 4).
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
    r"""Get the 3D rotation transformation matrix.

    Rotate v with axis n by an angle theta according to the right hand rule, follow 
    rodrigues' rotation formula.

    .. math::

        R(\mathbf{n}, \alpha) = \cos{\alpha} \cdot \mathbf{I}
        + (1-\cos{\alpha}) \cdot \mathbf{n} \cdot \mathbf{n^T}
        + \sin{\alpha} \cdot \mathbf{N} \\

        N  = \begin{pmatrix}
        0 & -n_z & n_y \\
        n_z & 0 & -n_x \\
        -n_y & n_x & 0
        \end{pmatrix}

    Args: 
        n: Rotation axis.
        theta: Rotation angle in radians.

    Returns:
        T: The homogeneous transformation matrix, shape (4, 4).
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
    """Get the 3D rotation transformation matrix.

    Rotate 3D vector `v` with `x`-axis by an angle theta according to the right
    hand rule.

    >>> np.allclose(
    ...     rotate3d_x(np.pi / 2),  # 90 degree rotation
    ...     [
    ...         [+1.0, +0.0, +0.0, +0.0],
    ...         [+0.0, +0.0, -1.0, +0.0],
    ...         [+0.0, +1.0, +0.0, +0.0],
    ...         [+0.0, +0.0, +0.0, +1.0],
    ...     ],
    ... )
    True

    Args:
        theta: float
            Rotation angle in radians.

    Returns:
        T: The homogeneous transformation matrix, shape (4, 4).
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
    """Get the 3D rotation transformation matrix.

    Rotate 3D vector `v` with `y`-axis by an angle theta according to the right
    hand rule.

    >>> np.allclose(
    ...     rotate3d_y(np.pi / 2),  # 90 degree rotation
    ...     [
    ...         [+0.0, +0.0, +1.0, +0.0],
    ...         [+0.0, +1.0, +0.0, +0.0],
    ...         [-1.0, +0.0, +0.0, +0.0],
    ...         [+0.0, +0.0, +0.0, +1.0],
    ...     ],
    ... )
    True

    Args:
        theta: Rotation angle in radians.

    Returns:
        T: The homogeneous transformation matrix, shape (4, 4).
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
    """Get the 3D rotation transformation matrix.

    Rotate 3D vector `v` with `z`-axis by an angle theta according to the right hand
    rule.

    >>> np.allclose(
    ...     rotate3d_z(np.pi / 2),  # 90 degree rotation
    ...     [
    ...         [+0.0, -1.0, +0.0, +0.0],
    ...         [+1.0, +0.0, +0.0, +0.0],
    ...         [+0.0, +0.0, +1.0, +0.0],
    ...         [+0.0, +0.0, +0.0, +1.0],
    ...     ],
    ... )
    True

    Args:
        theta: float
            Rotation angle in radians.

    Returns:
        T: np.NDArray
            The homogeneous transformation matrix, shape (4, 4).
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

    >>> np.allclose(to_homogeneous([1, 2, 3], 1), [1.0, 2.0, 3.0, 1.0])
    True
    >>> np.allclose(
    ...     to_homogeneous([[1, 2, 3], [4, 5, 6]], 0),
    ...     [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]],
    ... )
    True

    Args:
        xyz: Coordinate of shape (..., 3)
        w: w of homogeneous coordinate, 1 for dot, 0 for vector.

    Returns:
        xyz4: Array of shape (..., 4)
    """
    xyz = np.array(xyz)
    if xyz.ndim == 1:
        return _to_homogeneous(xyz[None, ...], w)[0]

    shape = xyz.shape[:-1]
    xyz = xyz.reshape(-1, xyz.shape[-1])
    xyz4 = _to_homogeneous(xyz, w).reshape(*shape, 4)
    return xyz4


def _to_homogeneous(xyz: npt.NDArray, w: float) -> npt.NDArray[np.float32]:
    """Fill xyz to homogeneous coordinates.

    Args:
        xyz: Coordinate of shape (N, 3)
        w: w of homogeneous coordinate, 1 for dot, 0 for vector.

    Returns:
        xyz4: Array of shape (N, 4)
    """
    if xyz.shape[1] == 4:
        return xyz

    assert xyz.shape[1] == 3
    filled = np.full((xyz.shape[0], 1), fill_value=w)
    xyz4 = np.concatenate([xyz, filled], axis=1)
    return xyz4


def model_view_transformation(
    position: Vec3f, look_at: Vec3f, up: Vec3f
) -> npt.NDArray[np.float32]:
    r"""Play model/view transformation.

    Args:
        position: Camera position \vec{e}.
        look_at: Camera look-at \vec{g}.
        up: Camera up direction \vec{t}.
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
    """Simple orthographic projection by drop z-axis

    >>> np.allclose(
    ...     orthographic_projection_simple(),
    ...     [
    ...         [1.0, 0.0, 0.0, 0.0],
    ...         [0.0, 1.0, 0.0, 0.0],
    ...         [0.0, 0.0, 0.0, 0.0],
    ...         [0.0, 0.0, 0.0, 0.0],
    ...     ],
    ... )
    True
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],  # drop z-axis
            [0, 0, 0, 0],  # drop w
        ],
        dtype=np.float32,
    )
