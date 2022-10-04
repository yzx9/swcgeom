"""Numpy related utils."""

from contextlib import contextmanager
import math
from typing import Any, Tuple

import numpy as np
import numpy.typing as npt

__all__ = ["padding1d", "XYPair", "to_distribution", "numpy_printoptions", "numpy_err"]


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

    dtype = dtype or (v is not None and v.dtype) or np.float32
    v = np.zeros(n, dtype=dtype) if v is None else v
    v = v.astype(dtype) if v.dtype != dtype else v
    assert v.ndim == 1

    if v.shape[0] >= n:
        return v[:n]

    padding = np.full(n - v.shape[0], padding_value, dtype=dtype)
    return np.concatenate([v, padding])


XYPair = Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]


def to_distribution(
    values: npt.NDArray,
    step: float,
    vmin: float = 0,
    vmax: float | None = None,
    norm: bool = False,
) -> XYPair:
    indices = np.floor(values / step - vmin).astype(np.int32)
    size_min = np.max(indices).item() + 1
    size = math.ceil((vmax - vmin) / step) + 1 if vmax is not None else size_min
    y = np.zeros(max(size, size_min), dtype=np.float32)
    for i in indices:
        y[i] = y[i] + 1

    if norm:
        y /= values.shape[0]

    x = vmin + step * np.arange(size, dtype=np.float32)
    return x, y[:size]


@contextmanager
def numpy_printoptions(*args, **kwargs):
    original_options = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)

    try:
        yield
    finally:
        np.set_printoptions(**original_options)


@contextmanager
def numpy_err(*args, **kwargs):
    old_settings = np.seterr(*args, **kwargs)

    try:
        yield
    finally:
        np.seterr(**old_settings)
