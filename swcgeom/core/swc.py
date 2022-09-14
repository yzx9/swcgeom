"""SWC format."""

from typing import Any, Iterable, TypeVar

import numpy as np
import numpy.typing as npt

__all__ = ["SWC", "SWCTypeVar"]


class SWC:
    """Abstract class that including swc infomation."""

    source: str | None

    def get_keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray[Any]:
        raise NotImplementedError()

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample,)."""
        return self.get_ndata("id")

    def type(self) -> npt.NDArray[np.int32]:
        """Get the types of shape (n_sample,)."""
        return self.get_ndata("type")

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x coordinates of shape (n_sample,)."""
        return self.get_ndata("x")

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y coordinates of shape (n_sample,)."""
        return self.get_ndata("y")

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z coordinates of shape (n_sample,)."""
        return self.get_ndata("z")

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of shape (n_sample,)."""
        return self.get_ndata("r")

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of parent of shape (n_sample,)."""
        return self.get_ndata("pid")

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the coordinates of shape(n_sample, 3)."""
        return np.stack([self.x(), self.y(), self.z()], axis=1)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the coordinates and radius array of shape(n_sample, 4)."""
        return np.stack([self.x(), self.y(), self.z(), self.r()], axis=1)


SWCTypeVar = TypeVar("SWCTypeVar", bound=SWC)
