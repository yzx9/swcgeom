"""Branch anlysis of tree."""

from functools import cached_property
from typing import List, TypeVar

import numpy as np
import numpy.typing as npt

from ..core import Branch, Tree

__all__ = ["BranchFeatures"]

T = TypeVar("T", bound=Branch)


class BranchFeatures:
    """Analysis bransh of tree."""

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_count(self) -> int:
        return len(self._branches)

    def get_length(self) -> npt.NDArray[np.float32]:
        """Get length of branches."""
        length = [br.length() for br in self._branches]
        return np.array(length, dtype=np.float32)

    def get_tortuosity(self) -> npt.NDArray[np.float32]:
        """Get tortuosity of path."""
        return np.array([br.tortuosity() for br in self._branches], dtype=np.float32)

    def get_angle(self, eps: float = 1e-7) -> npt.NDArray[np.float32]:
        """Get agnle between branches.

        Returns
        -------
        angle : npt.NDArray[np.float32]
            An array of shape (N, N), which N is length of branches.
        """

        return self.calc_angle(self._branches, eps=eps)

    @staticmethod
    def calc_angle(branches: List[T], eps: float = 1e-7) -> npt.NDArray[np.float32]:
        """Calc agnle between branches.

        Returns
        -------
        angle : npt.NDArray[np.float32]
            An array of shape (N, N), which N is length of branches.
        """

        vector = np.array([br[-1].xyz() - br[0].xyz() for br in branches])
        vector_dot = np.matmul(vector, vector.T)
        vector_norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
        vector_norm_dot = np.matmul(vector_norm, vector_norm.T) + eps
        arccos = np.clip(vector_dot / vector_norm_dot, -1, 1)
        angle = np.arccos(arccos)
        return angle

    @cached_property
    def _branches(self) -> List[Tree.Branch]:
        return self.tree.get_branches()
