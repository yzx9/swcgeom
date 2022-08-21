import copy
import itertools
from typing import Callable, overload

from .Branch import Branch
from .Tree import K, T, Tree


class BranchTree(Tree):
    """A branch tree that contains only soma, branch, and tip nodes."""

    class Node(Tree.Node):
        branches: list[Branch]

        def __init__(
            self, id: int, type: int, x: float, y: float, z: float, r: float, pid: int
        ) -> None:
            super().__init__(id, type, x, y, z, r, pid)
            self.branches = []

        @classmethod
        def from_parent(cls, p: Tree.Node) -> "BranchTree.Node":
            self = cls(p.id, p.type, p.x, p.y, p.z, p.r, p.pid)
            self.data = copy.copy(p.data)
            return self

    TraverseEnter = Callable[[Node, T | None], T]
    TraverseLeave = Callable[[Node, list[T]], T]

    @overload
    def traverse(self, enter: TraverseEnter[T]) -> None:
        ...

    @overload
    def traverse(
        self,
        enter: TraverseEnter[T] | None = None,
        leave: TraverseLeave[K] | None = None,
    ) -> K:
        ...

    def traverse(
        self,
        enter: TraverseEnter[T] | None = None,
        leave: TraverseLeave[K] | None = None,
    ) -> K | None:
        """Traverse each nodes.

        See Also
        --------
        Tree.traverse
        """
        return super().traverse(enter=enter, leave=leave)  # type: ignore

    def __getitem__(self, id: int) -> Node:
        return super().__getitem__(id)  # type: ignore

    @classmethod
    def from_swc(cls, swc_path: str) -> "BranchTree":
        """Generating a branch tree from swc file.

        Parameters
        ----------
        swc_path : str.
            file path of *.swc.

        Returns
        -------
        BranchTree : A branch tree.
        """

        tree = super().from_swc(swc_path)
        return cls.from_tree(tree)

    @classmethod
    def from_tree(cls, tree: Tree) -> "BranchTree":
        """Generating a branch tree from tree.

        Parameters
        ----------
        tree : Tree.
            A neuron tree.

        Returns
        -------
        BranchTree : A branch tree.
        """

        self = cls()
        self._source = tree._source

        def reducer(oldId: int, parentId: int | None) -> list[Tree.Node]:
            node = cls.Node.from_parent(tree[oldId])
            neighbors = list(tree.G.neighbors(oldId))
            if (parentId is not None) and (len(neighbors) == 1):
                branchNodes = reducer(neighbors[0], parentId)
                branchNodes.append(node)
                return branchNodes

            node.id = len(self.G) + 1
            node.pid = parentId if parentId is not None else -1
            self._add_node(node)
            if parentId is not None:
                self._add_edge(parentId, node.id)

            for n in neighbors:
                branchNodes = reducer(n, node.id)
                branchNodes.append(node)
                branchNodes.reverse()
                node.branches.append(Branch(branchNodes))

            return [node]

        reducer(self.root, None)
        return self

    def get_branches(self) -> list[Branch]:
        return self.traverse(
            leave=lambda n, acc: list(itertools.chain(n.branches, *acc))
        )
