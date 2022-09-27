"""Branch anlysis of tree."""

from typing import List

import numpy as np
import numpy.typing as npt

from ..core import Branch, BranchTree, Tree
from ..utils import to_distribution

__all__ = ["BranchAnalysis"]


class BranchAnalysis:
    """Analysis bransh of tree."""

    tree: Tree
    branch_tree: BranchTree
    branches: List[Branch]

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.branch_tree = BranchTree.from_tree(self.tree)
        self.branches = self.branch_tree.get_origin_branches()

    def get_length(self) -> npt.NDArray[np.float32]:
        """Get length of branches."""
        length = [br.length() for br in self.branches]
        return np.array(length, dtype=np.float32)

    def get_length_distribution(self, step: float = 1) -> npt.NDArray[np.int32]:
        """Get length distribution of branches."""
        lengths = self.get_length()
        return to_distribution(lengths, step)

    def get_angle(self, eps: float = 1e-7) -> npt.NDArray[np.float32]:
        """Get agnle between branches.

        Returns
        -------
        angle : npt.NDArray[np.float32]
            An array of shape (N, N), which N is length of branches.
        """

        vector = np.array([br[-1].xyz() - br[0].xyz() for br in self.branches])
        vector_dot = np.matmul(vector, vector.T)
        vector_norm = np.linalg.norm(vector, ord=2, axis=1, keepdims=True)
        vector_norm_dot = np.matmul(vector_norm, vector_norm.T) + eps
        arccos = np.clip(vector_dot / vector_norm_dot, -1, 1)
        angle = np.arccos(arccos)
        return angle
