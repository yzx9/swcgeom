"""The segment is a branch with two nodes."""

from typing import Dict, Generic, Iterable, List, TypeVar

import numpy as np
import numpy.typing as npt

from ..utils import padding1d
from .path import PathBase
from .swc import SWCTypeVar

__all__ = ["SegmentBase", "Segment", "SegmentAttached", "Segments"]


class SegmentBase(PathBase):
    r"""A segment is a branch with two nodes."""

    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray:
        raise NotImplementedError()


class Segment(SegmentBase):
    r"""A segment is a path with two nodes."""

    ndata: Dict[str, npt.NDArray]

    def __init__(
        self,
        *,
        type: npt.NDArray[np.int32] | None = None,  # pylint: disable=redefined-builtin
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        super().__init__()
        n_nodes = 2
        ndata = {
            "id": np.arange(0, n_nodes, step=1, dtype=np.int32),
            "type": padding1d(n_nodes, type, dtype=np.int32),
            "x": padding1d(n_nodes, x),
            "y": padding1d(n_nodes, y),
            "z": padding1d(n_nodes, z),
            "r": padding1d(n_nodes, r, padding_value=1),
            "pid": np.arange(-1, n_nodes - 1, step=1, dtype=np.int32),
        }
        kwargs.update(ndata)
        self.ndata = kwargs
        self.source = ""  # TODO

    def keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.ndata[key]


class SegmentAttached(SegmentBase, Generic[SWCTypeVar]):
    r"""Segment attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, pid: int, idx: int) -> None:
        super().__init__()
        self.attach = attach
        self.idx = np.array([pid, idx])

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def detach(self) -> Segment:
        return Segment(**{k: self[k] for k in self.keys()})


SegmentT = TypeVar("SegmentT", bound=SegmentBase)


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
