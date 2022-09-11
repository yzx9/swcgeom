"""Numpy related utils."""

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["padding1d"]


def padding1d(
    n: int,
    v: npt.NDArray | None,
    padding_value: Any = 0,
    dtype: npt.DTypeLike | None = None,
) -> npt.NDArray:
    """Padding x to array of shape (n,).

    Parameters
    ----------
    n : int
        Size of vector.
    v : np.ndarray, optional
        Input vector.
    padding_value : any, default to `0`.
        If x.shape[0] is less than n, the rest will be filled with
        padding value.
    dtype : np.DTypeLike, optional
        Data type of array. If specify, cast x to dtype, else dtype of
        x will used, otherwise defaults to `~numpy.float32`.
    """

    dtype = dtype or (v and v.dtype) or np.float32
    v = v or np.zeros(n, dtype=dtype)
    v = v if v.dtype != dtype else v.astype(dtype)
    assert v.ndim == 1

    if v.shape[0] >= n:
        return v[:n]

    padding = np.full(n - v.shape[0], padding_value, dtype=dtype)
    return np.concatenate([v, padding])
