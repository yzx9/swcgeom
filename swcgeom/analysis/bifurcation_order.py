"""Depth distribution of tree."""


from functools import cached_property

import numpy as np
import numpy.typing as npt

from ..core import BranchTree, Tree
from ..utils import to_distribution

__all__ = ["BifurcationOrderAnalysis"]


class BifurcationOrderAnalysis:
    """Calc bifurcation order of tree.

    Branch order is the number of bifurcations between current position
    and the root.
    """

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_bifurcation_order(self) -> npt.NDArray[np.int32]:
        """Get bifurcation order of tree.

        Returns
        -------
        order : npt.NDArray[np.int32]
            Array of shape (N,), while N is the number of nodes.
        """
        order = np.zeros_like(self._branch_tree.id(), dtype=np.int32)

        def assign_depth(n: Tree.Node, pre_depth: int | None) -> int:
            cur_order = pre_depth + 1 if pre_depth is not None else 0
            order[n.id] = cur_order
            return cur_order

        self._branch_tree.traverse(enter=assign_depth)
        return order

    def get_bifurcation_order_distribution(
        self,
        step: int = 1,
        /,
        filter_bifurcation: bool = False,
        filter_tip: bool = False,
        filter_other: bool = False,
    ) -> npt.NDArray[np.int32]:
        """Get bifurcation order distribution of tree.

        Parameters
        ----------
        filter_bifurcation : bool, default `False`
            Filter bifurcation nodes.
        filter_tip : bool, default `False`
            Filter tip nodes.
        filter_other : bool, default `False`
            Filter nodes that are not bifurcations or tips.
        """
        bifurcation_order = self.get_bifurcation_order()

        bifurcations = [n.id for n in self.tree.get_bifurcations()]
        tips = [n.id for n in self.tree.get_tips()]

        if filter_bifurcation:
            bifurcation_order[bifurcations] = -1

        if filter_tip:
            bifurcation_order[tips] = -1

        if filter_other:
            other = self.tree.id()
            other = np.setdiff1d(other, bifurcations)
            other = np.setdiff1d(other, tips)
            bifurcation_order[other] = -1

        bifurcation_order = bifurcation_order[bifurcation_order != -1]
        return to_distribution(bifurcation_order, step)

    @cached_property
    def _branch_tree(self) -> BranchTree:
        return BranchTree.from_tree(self.tree)
