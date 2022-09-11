"""Branch tree is simplified neuron tree."""

import itertools

from typing_extensions import Self  # TODO: move to typing in python 3.11

from .branch import Branch
from .swc_utils import from_swc
from .tree import Node, Tree


class BranchTree(Tree):
    """A branch tree that contains only soma, branch, and tip nodes."""

    def get_branches(self) -> list[Branch]:
        return self.traverse(
            leave=lambda n, p: list(itertools.chain(n["branches"], *p))
        )

    @classmethod
    def from_swc(cls, swc_path: str) -> Self:
        """Generating a branch tree from swc file.

        Parameters
        ----------
        swc_path : str
            Path of *.swc.

        Returns
        -------
        BranchTree
            A branch tree.
        """

        return cls.from_tree(from_swc(swc_path))

    @classmethod
    def from_tree(cls, tree: Tree) -> Self:
        """Generating a branch tree from tree.

        Parameters
        ----------
        tree : Tree
            A neuron tree.

        Returns
        -------
        BranchTree : BranchTree
            A branch tree.
        """

        self = cls()
        self.source = tree.source

        def reducer(old_id: int, parent_id: int | None) -> list[Node]:
            node = cls.Node(**tree[old_id])  # make shallow copy
            neighbors = list(tree.G.neighbors(old_id))
            if (parent_id is not None) and (len(neighbors) == 1):
                branch_nodes = reducer(neighbors[0], parent_id)
                branch_nodes.append(node)
                return branch_nodes

            node.id = len(self) + 1
            node.pid = parent_id if parent_id is not None else -1
            self._add_node(node)
            if parent_id is not None:
                self._add_edge(parent_id, node.id)

            for n in neighbors:
                branch_nodes = reducer(n, node.id)
                branch_nodes.append(node)
                branch_nodes.reverse()
                node.branches.append(Branch(branch_nodes))

            return [node]

        reducer(0, None)
        return self
