"""Depth distribution of tree."""


from functools import cached_property
from typing import List

import numpy as np
import numpy.typing as npt

from ..core import BranchTree, Tree
from ..utils import to_distribution, XYPair

__all__ = ["NodeFeatures"]


class NodeFeatures:
    """Calc node feature of tree."""

    tree: Tree

    def __init__(self, tree: Tree) -> None:
        self.tree = tree

    def get_radial_distance(self) -> npt.NDArray[np.float32]:
        """Get the end-to-end straight-line distance to soma.

        Returns
        -------
        radial_distance : npt.NDArray[np.float32]
            Array of shape (N,), while N is the number of nodes.
        """
        xyz = self.tree.xyz() - self.tree.soma().xyz()
        radial_distance = np.linalg.norm(xyz, axis=1)
        return radial_distance

    def get_radial_distance_distribution(
        self,
        step: float = 1,
        /,
        filter_bifurcation: bool = False,
        filter_tip: bool = False,
        filter_other: bool = True,
    ) -> XYPair:
        """Get radial distance distribution of tree.

        Parameters
        ----------
        filter_bifurcation : bool, default `False`
            Filter bifurcation nodes.
        filter_tip : bool, default `False`
            Filter tip nodes.
        filter_other : bool, default `False`
            Filter nodes that are not bifurcations or tips.

        Returns
        -------
        radial_distance : npt.NDArray[np.float32]
            Array of shape (N,), while N is the number of nodes.
        """
        return self._to_distribution(
            self.get_radial_distance(),
            step,
            filter_bifurcation=filter_bifurcation,
            filter_tip=filter_tip,
            filter_other=filter_other,
        )

    def get_branch_order(self) -> npt.NDArray[np.int32]:
        """Get branch order of tree.

        Bifurcation order is the number of bifurcations between current position
        and the root.

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

    def get_branch_order_distribution(
        self,
        step: int = 1,
        /,
        filter_bifurcation: bool = False,
        filter_tip: bool = False,
        filter_other: bool = True,
    ) -> XYPair:
        """Get branch order distribution of tree.

        Parameters
        ----------
        filter_bifurcation : bool, default `False`
            Filter bifurcation nodes.
        filter_tip : bool, default `False`
            Filter tip nodes.
        filter_other : bool, default `False`
            Filter nodes that are not bifurcations or tips.
        """
        return self._to_distribution(
            self.get_branch_order(),
            step,
            filter_bifurcation=filter_bifurcation,
            filter_tip=filter_tip,
            filter_other=filter_other,
        )

    def _to_distribution(
        self,
        x: npt.NDArray,
        step: float,
        /,
        filter_bifurcation: bool,
        filter_tip: bool,
        filter_other: bool,
    ) -> XYPair:
        if filter_bifurcation:
            x[self._bifurcations] = -1

        if filter_tip:
            x[self._tips] = -1

        if filter_other:
            x[self._other] = -1

        x = x[x != -1]
        return to_distribution(x, step)

    @cached_property
    def _branch_tree(self) -> BranchTree:
        return BranchTree.from_tree(self.tree)

    @cached_property
    def _bifurcations(self) -> List[int]:
        return [n.id for n in self.tree.get_bifurcations()]

    @cached_property
    def _tips(self) -> List[int]:
        return [n.id for n in self.tree.get_tips()]

    @cached_property
    def _other(self) -> npt.NDArray[np.int32]:
        other = self.tree.id()
        other = np.setdiff1d(other, self._bifurcations)
        other = np.setdiff1d(other, self._tips)
        return other
