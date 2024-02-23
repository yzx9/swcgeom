"""Branch is a set of node points."""

from typing import Generic, Iterable, List

import numpy as np
import numpy.typing as npt

from swcgeom.core.compartment import Compartment, Compartments
from swcgeom.core.path import Path
from swcgeom.core.swc import DictSWC, SWCTypeVar

__all__ = ["Branch"]


class Branch(Path, Generic[SWCTypeVar]):
    r"""Neural branch.

    Notes
    -----
    Only a part of data of branch nodes is valid, such as `x`, `y`, `z` and
    `r`, but the `id` and `pid` is usually invalid.
    """

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    class Compartment(Compartment["Branch"]):
        """Segment of branch."""

    Segment = Compartment  # Alias

    def __repr__(self) -> str:
        return f"Neuron branch with {len(self)} nodes."

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def get_compartments(self) -> Compartments[Compartment]:
        return Compartments(self.Compartment(self, n.pid, n.id) for n in self[1:])

    def get_segments(self) -> Compartments[Compartment]:
        return self.get_compartments()  # Alias

    def detach(self) -> "Branch[DictSWC]":
        """Detach from current attached object."""
        # pylint: disable=consider-using-dict-items
        attact = DictSWC(
            **{k: self[k] for k in self.keys()},
            source=self.attach.source,
            names=self.names,
        )
        attact.ndata[self.names.id] = self.id()
        attact.ndata[self.names.pid] = self.pid()
        return Branch(attact, self.id())

    @classmethod
    def from_xyzr(cls, xyzr: npt.NDArray[np.float32]) -> "Branch[DictSWC]":
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

        n_nodes = xyzr.shape[0]
        if xyzr.shape[1] == 3:
            ones = np.ones([n_nodes, 1], dtype=np.float32)
            xyzr = np.concatenate([xyzr, ones], axis=1)

        idx = np.arange(0, n_nodes, step=1, dtype=np.int32)
        attact = DictSWC(
            id=idx,
            type=np.full((n_nodes), fill_value=3, dtype=np.int32),
            x=xyzr[:, 0],
            y=xyzr[:, 1],
            z=xyzr[:, 2],
            r=xyzr[:, 3],
            pid=np.arange(-1, n_nodes - 1, step=1, dtype=np.int32),
        )
        return Branch(attact, idx)

    @classmethod
    def from_xyzr_batch(
        cls, xyzr_batch: npt.NDArray[np.float32]
    ) -> List["Branch[DictSWC]"]:
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

        branches: List[Branch[DictSWC]] = []
        for xyzr in xyzr_batch:
            n_nodes = xyzr.shape[0]
            idx = np.arange(0, n_nodes, step=1, dtype=np.int32)
            attact = DictSWC(
                id=idx,
                type=np.full((n_nodes), fill_value=3, dtype=np.int32),
                x=xyzr[:, 0],
                y=xyzr[:, 1],
                z=xyzr[:, 2],
                r=xyzr[:, 3],
                pid=np.arange(-1, n_nodes - 1, step=1, dtype=np.int32),
            )
            branches.append(Branch(attact, idx))

        return branches
