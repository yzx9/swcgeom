"""Branch tree is a simplified neuron tree."""

import itertools
from typing import Dict, List

from .branch import Branch
from .tree import Tree
from .tree_utils import to_sub_tree

__all__ = ["BranchTree"]


class BranchTree(Tree):
    """Branch tree keeps topology of tree.

    A branch tree that contains only soma, branch, and tip nodes.
    """

    branches: Dict[int, List[Branch]]

    def get_origin_branches(self) -> list[Branch]:
        """Get branches of original tree."""
        return list(itertools.chain(*self.branches.values()))

    def get_origin_node_branches(self, idx: int) -> List[Branch]:
        """Get branches of node of original tree."""
        return self.branches[idx]

    @staticmethod
    def from_tree(tree: Tree) -> "BranchTree":
        """Generating a branch tree from tree."""

        branches = tree.get_branches()

        sub_id = [br[-1].id for br in branches]
        sub_pid = [br[0].id for br in branches]
        # insert root
        sub_id.insert(0, 0)
        sub_pid.insert(0, -1)

        sub_tree, id_map = to_sub_tree(tree, sub_id, sub_pid)
        branch_tree = BranchTree(len(sub_tree), **sub_tree.ndata)
        branch_tree.source = tree.source  # TODO

        branch_tree.branches = {}
        for branch_raw in branches:
            idx = id_map[branch_raw[0].id]
            branch_tree.branches.setdefault(idx, [])
            branch_tree.branches[idx].append(branch_raw.detach())

        return branch_tree

    @staticmethod
    def from_swc(swc_file: str, **kwargs) -> "BranchTree":
        tree = super().from_swc(swc_file, **kwargs)
        return BranchTree.from_tree(tree)
