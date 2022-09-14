"""The segment is a branch with two nodes."""

from typing import Dict, Generic, Iterable

import numpy as np
import numpy.typing as npt

from ..utils import padding1d
from .node import Nodes
from .swc import SWCTypeVar

__all__ = ["Segment", "SegmentAttached"]


class Segment(Nodes):
    r"""A segment is a branch with two nodes."""

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
        self.source = None  # TODO

    def get_keys(self) -> Iterable[str]:
        return self.ndata.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.ndata[key]


class SegmentAttached(Nodes, Generic[SWCTypeVar]):
    """Segment attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, pid: int, idx: int) -> None:
        super().__init__()
        self.attach = attach
        self.idx = np.array([pid, idx])

    def get_keys(self) -> Iterable[str]:
        return self.attach.get_keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def detach(self) -> Segment:
        return Segment(**{k: self[k] for k in self.get_keys()})
