"""The segment is a branch with two nodes."""

from typing import Generic, Iterable, List, TypeVar

import numpy as np
import numpy.typing as npt

from .path import Path
from .swc import DictSWC, SWCTypeVar

__all__ = ["Segment", "Segments"]


class Segment(Path, Generic[SWCTypeVar]):
    r"""Segment attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, pid: int, idx: int) -> None:
        super().__init__(attach, np.array([pid, idx]))

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def detach(self) -> "Segment":
        """Detach from current attached object."""
        attact = DictSWC(**{k: self[k] for k in self.keys()})
        return Segment(attact, self.idx[0], self.idx[1])


SegmentT = TypeVar("SegmentT", bound=Segment)


class Segments(List[SegmentT]):
    r"""Segments contains a set of segments."""

    def __init__(self, segments: Iterable[SegmentT]) -> None:
        super().__init__(segments)

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample, 2)."""
        return self.get_ndata("id")

    def type(self) -> npt.NDArray[np.int32]:
        """Get the types of shape (n_sample, 2)."""
        return self.get_ndata("type")

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x coordinates of shape (n_sample, 2)."""
        return self.get_ndata("x")

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y coordinates of shape (n_sample, 2)."""
        return self.get_ndata("y")

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z coordinates of shape (n_sample, 2)."""
        return self.get_ndata("z")

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of shape (n_sample, 2)."""
        return self.get_ndata("r")

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of parent of shape (n_sample, 2)."""
        return self.get_ndata("pid")

    def xyz(self):
        """Get the coordinates of shape (n_sample, 2, 3)."""
        return np.stack([self.x(), self.y(), self.z()], axis=2)

    def xyzr(self):
        """Get the xyzr array of shape (n_sample, 2, 4)."""
        return np.stack([self.x(), self.y(), self.z(), self.r()], axis=2)

    def get_ndata(self, key: str) -> npt.NDArray:
        """Get ndata of shape (n_sample, 2).

        The order of axis 1 is (parent, current node).
        """
        return np.array([s.get_ndata(key) for s in self])
