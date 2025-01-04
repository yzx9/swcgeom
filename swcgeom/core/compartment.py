# Copyright 2022-2025 Zexin Yuan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""The segment is a branch with two nodes."""

from collections.abc import Iterable
from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt

from swcgeom.core.path import Path
from swcgeom.core.swc import DictSWC, SWCTypeVar
from swcgeom.core.swc_utils import SWCNames, get_names

__all__ = ["Compartment", "Compartments", "Segment", "Segments"]


class Compartment(Path, Generic[SWCTypeVar]):
    r"""Compartment attached to external object."""

    attach: SWCTypeVar
    idx: npt.NDArray[np.int32]

    def __init__(self, attach: SWCTypeVar, pid: int, idx: int) -> None:
        super().__init__(attach, np.array([pid, idx]))

    def keys(self) -> Iterable[str]:
        return self.attach.keys()

    def get_ndata(self, key: str) -> npt.NDArray:
        return self.attach.get_ndata(key)[self.idx]

    def detach(self) -> "Compartment[DictSWC]":
        """Detach from current attached object."""
        # pylint: disable=consider-using-dict-items
        attact = DictSWC(
            **{k: self[k] for k in self.keys()},
            source=self.attach.source,
            names=self.names,
        )
        attact.ndata[self.names.id] = self.id()
        attact.ndata[self.names.pid] = self.pid()
        return Compartment(attact, 0, 1)


T = TypeVar("T", bound=Compartment)


class Compartments(list[T]):
    r"""Comparments contains a set of comparment."""

    names: SWCNames

    def __init__(self, segments: Iterable[T]) -> None:
        super().__init__(segments)
        self.names = self[0].names if len(self) > 0 else get_names()

    def id(self) -> npt.NDArray[np.int32]:  # pylint: disable=invalid-name
        """Get the ids of shape (n_sample, 2)."""
        return self.get_ndata(self.names.id)

    def type(self) -> npt.NDArray[np.int32]:
        """Get the types of shape (n_sample, 2)."""
        return self.get_ndata(self.names.type)

    def x(self) -> npt.NDArray[np.float32]:
        """Get the x coordinates of shape (n_sample, 2)."""
        return self.get_ndata(self.names.x)

    def y(self) -> npt.NDArray[np.float32]:
        """Get the y coordinates of shape (n_sample, 2)."""
        return self.get_ndata(self.names.y)

    def z(self) -> npt.NDArray[np.float32]:
        """Get the z coordinates of shape (n_sample, 2)."""
        return self.get_ndata(self.names.z)

    def r(self) -> npt.NDArray[np.float32]:
        """Get the radius of shape (n_sample, 2)."""
        return self.get_ndata(self.names.r)

    def pid(self) -> npt.NDArray[np.int32]:
        """Get the ids of parent of shape (n_sample, 2)."""
        return self.get_ndata(self.names.pid)

    def xyz(self) -> npt.NDArray[np.float32]:
        """Get the coordinates of shape (n_sample, 2, 3)."""
        return np.stack([self.x(), self.y(), self.z()], axis=2)

    def xyzr(self) -> npt.NDArray[np.float32]:
        """Get the xyzr array of shape (n_sample, 2, 4)."""
        return np.stack([self.x(), self.y(), self.z(), self.r()], axis=2)

    def get_ndata(self, key: str) -> npt.NDArray:
        """Get ndata of shape (n_sample, 2).

        The order of axis 1 is (parent, current node).
        """
        return np.array([s.get_ndata(key) for s in self])


# Aliases
Segment = Compartment
Segments = Compartments
