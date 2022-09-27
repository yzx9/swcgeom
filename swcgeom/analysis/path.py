"""Depth distribution of tree."""


from typing import List
import numpy as np
import numpy.typing as npt

from ..core import Tree

__all__ = ["PathAnalysis"]


class PathAnalysis:
    """Path analysis of tree."""

    tree: Tree
    paths: List[Tree.Path] | None = None

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_count(self) -> int:
        return len(self._get_paths())

    def get_length(self) -> npt.NDArray[np.float32]:
        """Get length of paths."""
        length = [path.length() for path in self._get_paths()]
        return np.array(length, dtype=np.float32)

    def _get_paths(self) -> List[Tree.Path]:
        if self.paths is None:
            self.paths = self.tree.get_paths()

        return self.paths
