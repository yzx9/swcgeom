import itertools
from typing import Callable, Optional, overload

from typing_extensions import Self  # TODO: move to typing in python 3.11

from .Branch import Branch
from .Tree import K, T, Tree


class BranchTree(Tree):
    """A branch tree that contains only soma, branch, and tip nodes."""

    class Node(Tree.Node):
        # fmt:off
        @property
        def branches(self) -> list[Branch]: return self["branches"]
        @branches.setter
        def branches(self, v: list[Branch]): self["branches"] = v
        # fmt:on

        def __init__(
            self,
            id: int,
            type: int,
            x: float,
            y: float,
            z: float,
            r: float,
            pid: int,
            **kwargs,
        ) -> None:
            super().__init__(id, type, x, y, z, r, pid, branches=[], **kwargs)

    TraverseEnter = Callable[[Node, Optional[T]], T]
    TraverseLeave = Callable[[Node, list[T]], T]

    def get_branches(self) -> list[Branch]:
        return self.traverse(leave=lambda n, p: list(itertools.chain(n.branches, *p)))

    # fmt:off
    @overload
    def traverse(self, *, enter: TraverseEnter[T]) -> None: ...
    @overload
    def traverse(self, *, enter: Optional[TraverseEnter[T]] = None, leave: TraverseLeave[K]) -> K: ...
    # fmt:on

    def traverse(
        self,
        *,
        enter: Optional[TraverseEnter[T]] = None,
        leave: Optional[TraverseLeave[K]] = None,
    ) -> K | None:
        return super().traverse(enter=enter, leave=leave)  # type: ignore

    def __getitem__(self, id: int) -> Node:
        return super().__getitem__(id)  # type: ignore

    @classmethod
    def from_swc(cls, swc_path: str) -> Self:
        """Generating a branch tree from swc file.

        Parameters
        ==========
        swc_path : str.
            Path of *.swc.

        Returns
        =======
        BranchTree : BranchTree.
            A branch tree.
        """

        tree = super().from_swc(swc_path)
        return cls.from_tree(tree)

    @classmethod
    def from_tree(cls, tree: Tree) -> Self:
        """Generating a branch tree from tree.

        Parameters
        ==========
        tree : Tree.
            A neuron tree.

        Returns
        =======
        BranchTree : BranchTree.
            A branch tree.
        """

        self = cls()
        self._source = tree._source

        def reducer(oldId: int, parentId: Optional[int]) -> list[Tree.Node]:
            node = cls.Node(**tree[oldId])  # make shallow copy
            neighbors = list(tree.G.neighbors(oldId))
            if (parentId is not None) and (len(neighbors) == 1):
                branchNodes = reducer(neighbors[0], parentId)
                branchNodes.append(node)
                return branchNodes

            node.id = len(self) + 1
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
