"""Branch is a set of node points."""

from typing import Dict, Generic, Iterable

import numpy as np
import numpy.typing as npt
from typing_extensions import Self  # TODO: move to typing in python 3.11

from ..utils import padding1d
from .path import PathBase
from .segment import SegmentAttached, Segments
from .swc import SWCTypeVar

__all__ = ["BranchBase", "Branch", "BranchAttached"]


class BranchBase(PathBase):
    r"""Neuron branch."""

    class Segment(SegmentAttached["BranchBase"]):
        """Segment of branch."""

    def __repr__(self) -> str:
        return f"Neuron branch with {len(self)} nodes."

    def keys(self) -> Iterable[str]:
        raise NotImplementedError()

    def get_ndata(self, key: str) -> npt.NDArray:
        raise NotImplementedError()

    def get_segments(self) -> Segments[Segment]:
        return Segments([self.Segment(self, n.pid, n.id) for n in self[1:]])


class Branch(BranchBase):
    r"""A branch of neuron tree.

    Notes
    -----
    Only a part of data of branch nodes is valid, such as `x`, `y`, `z` and
    `r`, but the `id` and `pid` is usually invalid.
    """

    ndata: Dict[str, npt.NDArray]

    def __init__(
        self,
        n_nodes: int,
        *,
        type: npt.NDArray[np.int32] | None = None,  # pylint: disable=redefined-builtin
        x: npt.NDArray[np.float32] | None = None,
        y: npt.NDArray[np.float32] | None = None,
        z: npt.NDArray[np.float32] | None = None,
        r: npt.NDArray[np.float32] | None = None,
        **kwargs: npt.NDArray,
    ) -> None:
        super().__init__()
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

    @classmethod
    def from_xyzr(cls, xyzr: npt.NDArray[np.float32]) -> Self:
        r"""Create a branch from ~numpy.ndarray.

        Parameters
        ----------
        xyzr : npt.NDArray[np.float32]
            Collection of nodes. If shape (n, 4), both `x`, `y`, `z`, `r` of
            nodes is enabled. If shape (n, 3), only `x`, `y`, `z` is enabled
            and `r` will fill by 1.
        """
        assert xyzr.ndim == 2 and xyzr.shape[1] in (
            3,
            4,
        ), f"xyzr should be of shape (N, 3) or (N, 4), got {xyzr.shape}"

        if xyzr.shape[1] == 3:
            ones = np.ones([xyzr.shape[0], 1], dtype=np.float32)
            xyzr = np.concatenate([xyzr, ones], axis=1)

        return cls(
            xyzr.shape[0], x=xyzr[:, 0], y=xyzr[:, 1], z=xyzr[:, 2], r=xyzr[:, 3]
        )

    @classmethod
    def from_xyzr_batch(cls, xyzr_batch: npt.NDArray[np.float32]) -> list[Self]:
        r"""Create list of branch form ~numpy.ndarray.

        Parameters
        ----------
        xyzr : npt.NDArray[np.float32]
            Batch of collection of nodes. If shape (bs, n, 4), both `x`, `y`,
            `z`, `r` of nodes is enabled. If shape (bs, n, 3), only `x`, `y`,
            `z` is enabled and `r` will fill by 1.
        """

        assert xyzr_batch.ndim == 3
        assert xyzr_batch.shape[1] >= 3

        if xyzr_batch.shape[2] == 3:
            ones = np.ones(
                [xyzr_batch.shape[0], xyzr_batch.shape[1], 1], dtype=np.float32
            )
            xyzr_batch = np.concatenate([xyzr_batch, ones], axis=2)

        branches = list[Branch]()
        for xyzr in xyzr_batch:
            branch = cls(
                xyzr.shape[0],
                x=xyzr[:, 0],
                y=xyzr[:, 1],
                z=xyzr[:, 2],
                r=xyzr[:, 3],
            )
            branches.append(branch)

        return branches


class BranchAttached(BranchBase, Generic[SWCTypeVar]):
    r"""Branch attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, idx: npt.ArrayLike) -> None:
        super().__init__()
        self.attach = attach
        self.idx = np.array(idx, dtype=np.int32)

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def detach(self) -> Branch:
        return Branch(len(self), **{k: self[k] for k in self.keys()})
