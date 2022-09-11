"""Utils for branch."""

import numpy as np
import numpy.typing as npt

from .transforms import angle, rotate3d_x, rotate3d_y, rotate3d_z, scale3d, translate3d

__all__ = ["get_branch_standardize_matrix"]


def get_branch_standardize_matrix(
    xyz: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    r"""Get standarize transformation matrix.

    Standardized branch starts at (0, 0, 0), ends at (1, 0, 0), up at y.

    Parameters
    ----------
    xyz : np.ndarray[np.float32]
        The `x`, `y`, `z` matrix of shape (N, 3) of branch.

    Returns
    -------
    T : np.ndarray[np.float32]
        An homogeneous transfomation matrix of shape (4, 4).
    """

    assert xyz.ndim == 2
    assert xyz.shape[1] == 3

    xyz = xyz[:, :3]
    T = np.identity(4)
    v = np.concatenate([xyz[-1] - xyz[0], np.zeros((1))])[:, None]

    # translate to the origin
    T = translate3d(-xyz[0, 0], -xyz[0, 1], -xyz[0, 2]).dot(T)

    # scale to unit vector
    s = float(1 / np.linalg.norm(v[:3, 0]))
    T = scale3d(s, s, s).dot(T)

    # rotate v to the xz-plane, v should be (x, 0, z) now
    vy = np.dot(T, v)[:, 0]
    # when looking at the xz-plane along the positive y-axis, the
    # coordinates should be (z, x)
    T = rotate3d_y(angle([vy[2], vy[0]], [0, 1])).dot(T)

    # rotate v to the x-axis, v should be (1, 0, 0) now
    vx = np.dot(T, v)[:, 0]
    T = rotate3d_z(angle([vx[0], vx[1]], [1, 0])).dot(T)

    # rotate the farthest point to the xy-plane
    if xyz.shape[0] > 2:
        ones = np.ones([xyz.shape[0], 1])
        xyz4 = np.concatenate([xyz, ones], axis=1).transpose()  # (4, N)
        new_xyz4 = np.dot(T, xyz4)  # (4, N)
        max_index = np.argmax(np.linalg.norm(new_xyz4[1:3, :], axis=0)[1:-1]) + 1
        max_xyz4 = xyz4[:, max_index].reshape(4, 1)
        max_xyz4_t = np.dot(T, max_xyz4)  # (4, 1)
        angle_x = angle(max_xyz4_t[1:3, 0], [1, 0])
        T = rotate3d_x(angle_x).dot(T)

    return T
