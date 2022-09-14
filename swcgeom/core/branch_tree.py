"""Branch tree is a simplified neuron tree."""

import itertools
from typing import Dict, List

from typing_extensions import Self  # TODO: move to typing in python 3.11

from .branch import Branch
from .swc_utils import to_sub_tree
from .tree import Tree


class BranchTree(Tree):
    """A branch tree that contains only soma, branch, and tip nodes."""

    branches: Dict[int, List[Branch]]

    def get_branches(self) -> list[Branch]:
        return list(itertools.chain(*self.branches.values()))

    def get_node_branches(self, idx: int) -> List[Branch]:
        return self.branches[idx]

    @classmethod
    def from_tree(cls, tree: Tree) -> Self:
        """Generating a branch tree from tree."""

        branches = tree.get_branches()

        sub_id = [br[-1].id for br in branches]
        sub_pid = [br[0].id for br in branches]
        # insert root
        sub_id.insert(0, 0)
        sub_pid.insert(0, -1)

        sub_tree, id_map = to_sub_tree(tree, sub_id, sub_pid)
        ndata = {k: sub_tree.get_ndata(k) for k in sub_tree.get_keys()}
        self = cls(len(sub_tree), **ndata)
        self.source = tree.source

        self.branches = {}
        for branch_raw in branches:
            idx = id_map[branch_raw[0].id]
            self.branches.setdefault(idx, [])
            self.branches[idx].append(branch_raw.detach())

        return self
