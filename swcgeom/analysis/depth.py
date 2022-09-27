"""Depth distribution of tree."""


from typing import List

import numpy as np
import numpy.typing as npt

from ..core import BranchTree, Tree

__all__ = ["DepthAnalysis"]


class DepthAnalysis:
    """Calc depth distribution of tree."""

    tree: Tree
    branch_tree: BranchTree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree
        self.branch_tree = BranchTree.from_tree(self.tree)

    def get_branch_depth(self) -> npt.NDArray[np.int32]:
        """Get depth distribution of branches."""
        depth = np.zeros_like(self.branch_tree.id())
        count = {}

        def assign_depth(n: Tree.Node, pre_depth: int | None) -> int:
            order = pre_depth + 1 if pre_depth is not None else 0
            depth[n.id] = order
            return order

        def collect_depth(n: Tree.Node, children: List[None]) -> None:
            if len(children) != 0:  # ignore tips
                count[depth[n.id]] = count.get(depth[n.id], 0) + 1

        self.branch_tree.traverse(enter=assign_depth, leave=collect_depth)
        distribution = [count.get(i, 0) for i in range(max(count.keys()))]
        return np.array(distribution, dtype=np.int32)

    def get_tip_depth(self) -> npt.NDArray[np.int32]:
        """Get depth distribution of tips."""
        depth = np.zeros_like(self.branch_tree.id())
        count = {}

        def assign_depth(n: Tree.Node, pre_depth: int | None) -> int:
            order = pre_depth + 1 if pre_depth is not None else 0
            depth[n.id] = order
            return order

        def collect_depth(n: Tree.Node, children: List[None]) -> None:
            if len(children) == 0:  # only tips
                count[depth[n.id]] = count.get(depth[n.id], 0) + 1

        self.branch_tree.traverse(enter=assign_depth, leave=collect_depth)
        distribution = [count.get(i, 0) for i in range(max(count.keys()))]
        return np.array(distribution, dtype=np.int32)
