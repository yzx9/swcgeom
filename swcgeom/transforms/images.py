"""Image stack related transform."""


from typing import Tuple

import numpy as np
import numpy.typing as npt

from .base import Transform

__all__ = ["Center"]


class Center(Transform[npt.NDArray[np.float32], npt.NDArray[np.float32]]):
    """Get image stack center."""

    def __init__(self, shape_out: int | Tuple[int, int, int]):
        super().__init__()
        self.shape_out = (
            shape_out
            if isinstance(shape_out, tuple)
            else (shape_out, shape_out, shape_out)
        )

    def __call__(self, x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        diff = np.subtract(x.shape[:3], self.shape_out)
        s = diff // 2
        e = s + x.shape[:3]
        return x[s[0] : e[0], s[1] : e[1], s[2] : e[2], :]

    def __repr__(self) -> str:
        return f"Center-{self.shape_out[0]}-{self.shape_out[1]}-{self.shape_out[2]}"
