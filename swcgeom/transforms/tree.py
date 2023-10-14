"""Transformation in tree."""

import warnings
from typing import Callable, List, Optional, Tuple

import numpy as np

from swcgeom.core import BranchTree, DictSWC, Path, Tree, cut_tree, to_subtree
from swcgeom.core.swc_utils import SWCTypes, get_types
from swcgeom.transforms.base import Transform
from swcgeom.transforms.branch import BranchConvSmoother
from swcgeom.transforms.geometry import Normalizer

__all__ = [
    "ToBranchTree",
    "ToLongestPath",
    "TreeSmoother",
    "TreeNormalizer",
    "CutByType",
    "CutAxonTree",
    "CutDendriteTree",
    "CutByBifurcationOrder",
    "CutShortTipBranch",
]


# pylint: disable=too-few-public-methods
class ToBranchTree(Transform[Tree, BranchTree]):
    """Transform tree to branch tree."""

    def __call__(self, x: Tree) -> BranchTree:
        return BranchTree.from_tree(x)


class ToLongestPath(Transform[Tree, Path[DictSWC]]):
    """Transform tree to longest path."""

    def __init__(self, *, detach: bool = True) -> None:
        self.detach = detach

    def __call__(self, x: Tree) -> Path[DictSWC]:
        paths = x.get_paths()
        idx = np.argmax([p.length() for p in paths])
        path = paths[idx]
        if self.detach:
            path = path.detach()
        return path  # type: ignore


class TreeSmoother(Transform[Tree, Tree]):  # pylint: disable=missing-class-docstring
    def __init__(self, n_nodes: int = 5) -> None:
        super().__init__()
        self.n_nodes = n_nodes
        self.trans = BranchConvSmoother(n_nodes=n_nodes)

    def __call__(self, x: Tree) -> Tree:
        x = x.copy()
        for br in x.get_branches():
            # TODO: works but is weird
            smoothed = self.trans(br)
            x.ndata["x"][br.origin_id()] = smoothed.x()
            x.ndata["y"][br.origin_id()] = smoothed.y()
            x.ndata["z"][br.origin_id()] = smoothed.z()

        return x

    def __repr__(self) -> str:
        return f"TreeSmoother-{self.n_nodes}"


class TreeNormalizer(Normalizer[Tree]):
    """Noramlize coordinates and radius to 0-1."""

    def __init__(self, *args, **kwargs) -> None:
        warnings.warn(
            "`TreeNormalizer` has been replaced by `Normalizer` since "
            "v0.6.0 beacuse it applies more widely, and this will be "
            "removed in next version",
            DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class CutByType(Transform[Tree, Tree]):
    """Cut tree by type.

    In order to preserve the tree structure, all ancestor nodes of the node to be preserved will be preserved.

    Notes
    -----
    Not all reserved nodes are of the specified type.
    """

    def __init__(self, type: int) -> None:  # pylint: disable=redefined-builtin
        super().__init__()
        self.type = type

    def __call__(self, x: Tree) -> Tree:
        removals = set(x.id()[x.type() != self.type])

        def leave(n: Tree.Node, keep_children: List[bool]) -> bool:
            if n.id in removals and any(keep_children):
                removals.remove(n.id)
            return n.id not in removals

        x.traverse(leave=leave)
        y = to_subtree(x, removals)
        return y

    def __repr__(self) -> str:
        return f"CutByType-{self.type}"


class CutAxonTree(CutByType):
    """Cut axon tree."""

    def __init__(self, types: Optional[SWCTypes] = None) -> None:
        types = get_types(types)
        super().__init__(type=types.axon)

    def __repr__(self) -> str:
        return "CutAxonTree"


class CutDendriteTree(CutByType):
    """Cut dendrite tree."""

    def __init__(self, types: Optional[SWCTypes] = None) -> None:
        types = get_types(types)
        super().__init__(type=types.basal_dendrite)  # TODO: apical dendrite

    def __repr__(self) -> str:
        return "CutDenriteTree"


class CutByBifurcationOrder(Transform[Tree, Tree]):
    """Cut tree by bifurcation order."""

    max_bifurcation_order: int

    def __init__(self, max_bifurcation_order: int) -> None:
        self.max_bifurcation_order = max_bifurcation_order

    def __call__(self, x: Tree) -> Tree:
        return cut_tree(x, enter=self._enter)

    def __repr__(self) -> str:
        return f"CutByBifurcationOrder-{self.max_bifurcation_order}"

    def _enter(self, n: Tree.Node, parent_level: int | None) -> Tuple[int, bool]:
        if parent_level is None:
            level = 0
        elif n.is_bifurcation():
            level = parent_level + 1
        else:
            level = parent_level
        return (level, level >= self.max_bifurcation_order)


class CutShortTipBranch(Transform[Tree, Tree]):
    """Cut off too short terminal branches.

    This method is usually applied in the post-processing of manual
    reconstruction. When the user draw lines, a line head is often left
    at the junction of two lines.
    """

    thre: float
    callbacks: List[Callable[[Tree.Branch], None]]

    def __init__(
        self, thre: float = 5, callback: Optional[Callable[[Tree.Branch], None]] = None
    ) -> None:
        self.thre = thre
        self.callbacks = []

        if callback is not None:
            self.callbacks.append(callback)

    def __repr__(self) -> str:
        return f"CutShortTipBranch-{self.thre}"

    def __call__(self, x: Tree) -> Tree:
        removals: List[int] = []
        self.callbacks.append(lambda br: removals.append(br[1].id))
        x.traverse(leave=self._leave)
        self.callbacks.pop()
        return to_subtree(x, removals)

    def _leave(
        self, n: Tree.Node, children: List[Tuple[float, Tree.Node] | None]
    ) -> Tuple[float, Tree.Node] | None:
        if len(children) == 0:  # tip
            return 0, n

        if len(children) == 1 and children[0] is not None:  # elongation
            dis, child = children[0]
            dis += n.distance(child)
            return dis, n

        for c in children:
            if c is None:
                continue

            dis, child = c
            if dis + n.distance(child) > self.thre:
                continue

            path = [n.id]  # n does not delete, but will include in callback
            while child is not None:  # TODO: perf
                path.append(child.id)
                child = cc[0] if len((cc := child.children())) > 0 else None

            br = Tree.Branch(n.attach, path)
            for cb in self.callbacks:
                cb(br)

        return None
