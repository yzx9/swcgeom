"""Image stack related transform."""


from typing import Tuple

import numpy as np
import numpy.typing as npt

from swcgeom.transforms.base import Transform

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
        e = np.add(s, self.shape_out)
        return x[s[0] : e[0], s[1] : e[1], s[2] : e[2], :]

    def extra_repr(self) -> str:
        return f"shape_out=({','.join(str(a) for a in self.shape_out)})"
