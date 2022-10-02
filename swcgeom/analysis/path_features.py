"""Depth distribution of tree."""


from functools import cached_property
from typing import List

import numpy as np
import numpy.typing as npt

from ..core import Tree

__all__ = ["PathFeatures"]


class PathFeatures:
    """Path analysis of tree."""

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_count(self) -> int:
        return len(self._paths)

    def get_length(self) -> npt.NDArray[np.float32]:
        """Get length of paths."""
        length = [path.length() for path in self._paths]
        return np.array(length, dtype=np.float32)

    def get_tortuosity(self) -> npt.NDArray[np.float32]:
        """Get tortuosity of path."""
        return np.array([path.tortuosity() for path in self._paths], dtype=np.float32)

    @cached_property
    def _paths(self) -> List[Tree.Path]:
        return self.tree.get_paths()
