"""Depth distribution of tree."""


from functools import cached_property
from typing import List

import numpy as np
import numpy.typing as npt

from ..core import Tree
from ..utils import to_distribution

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

    def get_length_distribution(self, step: float = 10) -> npt.NDArray[np.int32]:
        """Get length distribution of paths."""
        return to_distribution(self.get_length(), step)

    def get_tortuosity(self) -> npt.NDArray[np.float32]:
        """Get tortuosity of path."""
        return np.array([path.tortuosity() for path in self._paths], dtype=np.float32)

    def get_tortuosity_distribution(self, step: float = 0.1) -> npt.NDArray[np.int32]:
        """Get tortuosity distribution of path."""
        return to_distribution(self.get_tortuosity(), step)

    @cached_property
    def _paths(self) -> List[Tree.Path]:
        return self.tree.get_paths()
